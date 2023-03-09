import numpy as np

class HodgkinHuxley:
    """Full Hodgkin-Huxley Model implemented in Python"""
    def __init__(self, dt=0.01, Cm=1.0, gNa=120.0, gK=36.0, gL=0.3, ENa=50.0, EK=-77.0, EL=-54.387):
        self.dt = dt  # デルタ
        self.Cm = Cm  # 膜容量(uF/cm^2)
        self.gNa = gNa  # Na+ の最大コンダクタンス(mS/cm^2)
        self.gK = gK  # K+ の最大コンダクタンス(mS/cm^2)
        self.gL = gL  # 漏れイオンの最大コンダクタンス(mS/cm^2)
        self.ENa = ENa  # Na+ の平衡電位(mV)
        self.EK = EK  # K+ の平衡電位(mV)
        self.EL = EL  # 漏れイオンの平衡電位(mV)

        self.V = -65.0  # 静止膜電位は任意の値をとれる. -Vを-(V+65)に変更
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32

    def alpha_m(self):
        return 0.1 * (self.V + 40.0) / (1.0 - np.exp(-(self.V + 40.0) / 10.0))

    def beta_m(self):
        return 4.0 * np.exp(-(self.V + 65.0) / 18.0)

    def alpha_h(self):
        return 0.07 * np.exp(-(self.V + 65.0) / 20.0)

    def beta_h(self):
        return 1.0 / (1.0 + np.exp(-(self.V + 35.0) / 10.0))

    def alpha_n(self):
        return 0.01 * (self.V + 55.0) / (1.0 - np.exp(-(self.V + 55.0) / 10.0))

    def beta_n(self):
        return 0.125 * np.exp(-(self.V + 65.0) / 80.0)

    def INa(self):
        return self.gNa * self.m ** 3 * self.h * (self.V - self.ENa)

    def IK(self):
        return self.gK * self.n ** 4 * (self.V - self.EK)

    def IL(self):
        return self.gL + (self.V - self.EL)

    def step(self, I_inj=0):
        """
         1step実行
         Args: I_inj:外部からの入力電流

         Returns: チャネル変数:m,h,n 膜電位:V
         """
        self.m += (self.alpha_m() * (1.0 - self.m) - self.beta_m() * self.m) * self.dt
        self.h += (self.alpha_h() * (1.0 - self.h) - self.beta_h() * self.h) * self.dt
        self.n += (self.alpha_n() * (1.0 - self.n) - self.beta_n() * self.n) * self.dt
        self.V += ((I_inj - self.INa() - self.IK() - self.IL()) / self.Cm) * self.dt
        return self.m, self.h, self.n, self.V