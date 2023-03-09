import numpy as np
import matplotlib.pyplot as plt

from HH_package.hodgkinhuxley import HodgkinHuxley

"""
観測ノイズの標準偏差
"""
SIGMA = 5.0

class ParticleFilter8(HodgkinHuxley):
    """
    Prticle filter implemented in python.
    """
    def __init__(self, V_train, V_test, m_test, h_test, n_test, n_particle=100, dt=0.05, C_m=1.0, g_Na=120.0, g_K=36.0,
                 g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.387):
        self.V_train = V_train  # 観測データ
        # 真のデータ
        self.V_test = V_test
        self.m_test = m_test
        self.h_test = h_test
        self.n_test = n_test
        self.n_particle = n_particle
        super().__init__(dt, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L)
        self.dim_param = 4

        self.T = len(self.V_test)  # 時系列データ数
        # 粒子フィルタ用 (T * n_particle)
        self.X_p = np.zeros((self.dim_param, self.T + 1, self.n_particle))
        # 粒子フィルタ.リサンプリング用
        self.X_p_resampled = np.zeros((self.dim_param, self.T + 1, self.n_particle))
        # 尤度(test data との誤差)
        self.w_V = np.zeros((self.T, self.n_particle))
        # 尤度を正規化したもの
        self.w_V_normed = np.zeros((self.T, self.n_particle))
        # 平均値
        self.m_average = np.zeros((self.T))
        self.h_average = np.zeros((self.T))
        self.n_average = np.zeros((self.T))
        self.V_average = np.zeros((self.T))

    def norm_likelihood(self, y, x):
        """
        尤度を計算.
        Args: y: 観測値 ,x:一期先予測した粒子
        Returns: exp(-(x-y)^2)
        """
        return np.exp(-(x - y) ** 2)

    def F_inv(self, w_cumsum, idx, u):
        """
        乱数を生成し, その時の粒子の番号を返す.

        Args: w_cumsum(array): 正規化した尤度の累積和, idx(array):[0,~,99](n=100)のarray, u(float):0~1の乱数）

        Returns: k+1: 選択された粒子の番号
        """
        if not np.any(w_cumsum < u):  # 乱数uがw_cumsumのどの値よりも小さいとき0を返す
            return 0
        k = np.max(idx[w_cumsum < u])  # uがwより大きいもので最大のものを返す
        return k + 1

    def resampling(self, weights):
        """
        リサンプリングを行う.

        Args: weights(array): 正規化した尤度の配列

        Returns: k_list: リサンプリングされて選択された粒子の番号の配列
        """
        w_cumsum = np.cumsum(weights)  # 正規化した重みの累積和をとる
        idx = np.asanyarray(range(self.n_particle))  # -> [0,1,2,,,98,99] (n=100)
        k_list = np.zeros(self.n_particle, dtype=np.int32)  # サンプリングしたｋのリスト格納

        # 乱数を粒子数つくり、関数を用いてk_listに値を格納
        for i, u in enumerate(np.random.random_sample(self.n_particle)):
            k = self.F_inv(w_cumsum, idx, u)
            k_list[i] = k
        return k_list

    def simulate(self):
        """
        粒子フィルタの実行
        """
        # 初期値設定
        initial_V_p = np.random.normal(-65, 1.0, self.n_particle)
        initial_m_p = np.random.normal(0.03, 0.005, self.n_particle)
        initial_h_p = np.random.normal(0.6, 0.01, self.n_particle)
        initial_n_p = np.random.normal(0.32, 0.01, self.n_particle)
        self.X_p[0, 0] = initial_m_p
        self.X_p[1, 0] = initial_h_p
        self.X_p[2, 0] = initial_n_p
        self.X_p[3, 0] = initial_V_p
        self.X_p_resampled[0, 0] = initial_m_p
        self.X_p_resampled[1, 0] = initial_h_p
        self.X_p_resampled[2, 0] = initial_n_p
        self.X_p_resampled[3, 0] = initial_V_p

        # 外部電流
        I_inj = np.zeros(self.T)
        I_inj[:] = 20

        for t in range(self.T):
            for i in range(self.n_particle):
                # HHモデルのステップに入力する値を設定
                self.m = self.X_p_resampled[0, t, i]
                self.h = self.X_p_resampled[1, t, i]
                self.n = self.X_p_resampled[2, t, i]
                self.V = self.X_p_resampled[3, t, i]
                # step関数を実行して, t → t+1へ遷移
                noise = np.random.normal(0, 0.1)  # ノイズあり
                noise_m = np.random.normal(0, 0.001)
                result = self.step(I_inj[t])
                self.X_p[0, t + 1, i] = result[0] + noise_m
                self.X_p[1, t + 1, i] = result[1] + noise_m
                self.X_p[2, t + 1, i] = result[2] + noise_m
                self.X_p[3, t + 1, i] = result[3] + noise
                # 尤度をテストデータを用いて計算
                self.w_V[t, i] = self.norm_likelihood(self.V_train[t], self.X_p[3, t + 1, i])
            # 求めた尤度を正規化e
            self.w_V_normed[t] = self.w_V[t] / np.sum(self.w_V[t])
            # 　リサンプリングを行う
            k_V = self.resampling(self.w_V_normed[t])
            self.X_p_resampled[:, t + 1] = self.X_p[:, t + 1, k_V]
            # リサンプリングした粒子の平均値を計算
            self.m_average[t] = np.sum(self.X_p_resampled[0, t + 1]) / self.n_particle
            self.h_average[t] = np.sum(self.X_p_resampled[1, t + 1]) / self.n_particle
            self.n_average[t] = np.sum(self.X_p_resampled[2, t + 1]) / self.n_particle
            self.V_average[t] = np.sum(self.X_p_resampled[3, t + 1]) / self.n_particle

    def draw_graph(self):
        # グラフ描画
        plt.figure(figsize=(10, 6))
        # plt.plot(range(self.T), self.V_train, label='train data')
        plt.plot(range(self.T), self.V_test, linewidth=3, linestyle="dashed", label='V: true')
        plt.plot(range(self.T), self.V_average, label='V particle average')
        # plt.title('particle filter')
        plt.xlabel('t[ms]')
        plt.ylabel('V[mV]')
        plt.grid()
        plt.legend()
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.T), self.m_test, linewidth=3, linestyle="dashed", label='m: truth')
        plt.plot(range(self.T), self.h_test, linewidth=3, linestyle="dashed", label='h: truth')
        plt.plot(range(self.T), self.n_test, linewidth=3, linestyle="dashed", label='n: truth')
        plt.plot(range(self.T), self.m_average, c='b', label='m: particle average')
        plt.plot(range(self.T), self.h_average, c='r', label='h: particle average')
        plt.plot(range(self.T), self.n_average, c='g', label='n: particle average')
        # plt.title('particle filter')
        plt.xlabel('t[ms]')
        plt.ylabel('V[mV]')
        plt.grid()
        plt.legend(loc='upper right')
        plt.show()

def generate_train_data():
    """
    テストデータを作成
    """
    # HHモデルをインスタンス化
    HH = HodgkinHuxley(dt=0.05)
    # steps数を定義
    steps = 1000
    t = np.arange(0, steps)
    # 外部電流
    I_inj = np.zeros(steps)
    I_inj[:] = 20
    # データの作成
    m_test = np.zeros(steps)
    h_test = np.zeros(steps)
    n_test = np.zeros(steps)
    V_test = np.zeros(steps)
    V_train = np.zeros(steps)
    for index, i in enumerate(I_inj):
        # システムノイズ
        noise_v = np.random.normal(0, 0.1)
        result = HH.step(i)
        m_test[index] = result[0]
        h_test[index] = result[1]
        n_test[index] = result[2]
        V_test[index] = result[3]
        V_train[index] = result[3] + noise_v
    # 観測ノイズ
    noise_observation = np.random.normal(0, SIGMA, steps)
    V_train += noise_observation
    return V_train, V_test, m_test, h_test, n_test

def main():
    test_data = generate_train_data()
    V_train = test_data[0]
    V_test = test_data[1]
    m_test = test_data[2]
    h_test = test_data[3]
    n_test = test_data[4]
    pf = ParticleFilter8(V_train, V_test, m_test, h_test, n_test)
    pf.simulate()
    pf.draw_graph()

if __name__ == '__main__':
    main()