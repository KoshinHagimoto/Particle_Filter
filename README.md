paticle_filter(粒子フィルタ・逐次モンテカルロ法)を用いて, Hodgkin-Huxleyモデルを用いて作成した観測膜電位データから真の膜電位とチャネル変数m,h,nの推定を行う.


hodgkinhuxley.py : シミュレーションデータ作成するためのHHモデルのファイル

particle_filter.py : 観測可能なノイズありの観測膜電位(V_train)から4次元の潜在変数(V,m,h,n)を粒子フィルタアルゴリズムを用いて推定を行う.
