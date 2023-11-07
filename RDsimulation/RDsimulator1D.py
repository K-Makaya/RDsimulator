#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from matrix_visualizer import MatrixVisualizer
import matplotlib.pyplot as plt
import sympy as sym

# visualizerの初期化。表示領域のサイズを与える。
visualizer = MatrixVisualizer(600, 600)

# シミュレーションの各パラメタ
VISUALIZATION_TIME = 1  # size of visualized time duration = visualization height
SPACE_SIZE = 256  # size of 1D space = visualization width
dx = 0.01
dt = 0.1
visualization_step = 100

# モデルの各パラメタ
Du = 2e-5
Dv = 1e-5
f, k = 0.04, 0.06

# 初期化
u = np.zeros((VISUALIZATION_TIME, SPACE_SIZE))
v = np.zeros((VISUALIZATION_TIME, SPACE_SIZE))
INIT_PATTERN_SIZE = 20
u[0,:] = 1.0
v[0,:] = 0.0
u[0, SPACE_SIZE//2-INIT_PATTERN_SIZE//2:SPACE_SIZE//2+INIT_PATTERN_SIZE//2] = 0.5
v[0, SPACE_SIZE//2-INIT_PATTERN_SIZE//2:SPACE_SIZE//2+INIT_PATTERN_SIZE//2] = 0.25
# 対称性を壊すために、少しノイズを入れる
u[0,:] += np.random.rand(SPACE_SIZE)*0.01
v[0,:] += np.random.rand(SPACE_SIZE)*0.01

t = 0
X = np.arange(SPACE_SIZE)
def d_u(u, v):
    return -u*v*v + f*(1-u)

def d_v(u, v):
    return u*v*v - v*(f+k)

def fitzhugh(state, t):
    u, v = state
    deltau = d_u(u,v)
    deltav = d_v(u,v)
    return deltau, deltav

fig, ax = plt.subplots(1,2,figsize=(16,8))

ax[0].set_ylim(0, 1.0)
ax[0].set_ylabel("u,v", size=20)
ax[0].set_title("u v concentration", size=20)
ax[1].set_xlim(0, 1.0)
ax[1].set_ylim(0, 1.0)
ax[1].set_xlabel("u", size=20)
ax[1].set_ylabel("v", size=20)
ax[1].set_title("nullcline", size=20)
U1, V1 = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
U2, V2 = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
dU1 = d_u(U1, V1)
dV1 = d_v(U1, V1)
dU2 = d_u(U2, V2)
dV2 = d_v(U2, V2)
ax[1].quiver(U2, V2, dU2, dV2)
ax[1].contour(U1, V1, dV1, levels=[0], colors="green")
ax[1].contour(U1, V1, dU1, levels=[0], colors="red")
ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 1])
ax[1].grid()

while visualizer:  # visualizerはウィンドウが閉じられるとFalseを返す
    for i in range(visualization_step):
        current_line = (t * visualization_step + i) % VISUALIZATION_TIME
        next_line = (current_line + 1) % VISUALIZATION_TIME
        current_u = u[current_line]
        current_v = v[current_line]
        # ラプラシアンの計算
        laplacian_u = (np.roll(current_u, 1) + np.roll(current_u, -1) - 2*current_u) / (dx*dx)
        laplacian_v = (np.roll(current_v, 1) + np.roll(current_v, -1) - 2*current_v) / (dx*dx)
        # Gray-Scottモデル方程式
        dudt = Du*laplacian_u - current_u*current_v*current_v + f*(1.0-current_u)
        dvdt = Dv*laplacian_v + current_u*current_v*current_v - (f+k)*current_v
        u[next_line] = current_u + dt * dudt
        v[next_line] = current_v + dt * dvdt
        t += 1

    # 表示をアップデート。
    line1,= ax[0].plot(X, u[0], color='#9999ff')
    line2,= ax[0].plot(X, v[0], color='#ff9999')
    
    point = ax[1].scatter(u[0],v[0],color="blue")

    visualizer.update(u)

    plt.pause(0.0005)
    point.remove()
    line1.remove()
    line2.remove()
    