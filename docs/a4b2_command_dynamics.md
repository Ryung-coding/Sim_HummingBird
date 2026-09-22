# HB A4B2 differential allocation 검증

## 범위와 결론

이 문서는 Hexa가 아니라 HummingBird(HB)의 A4B2 allocator만 다룬다.

핵심 결론은 다음과 같다.

1. `W_INV_f`를 바꾸더라도 full-row-rank인 `J`에서 포화와 모델 오차가 없다면 primary wrench-rate는 바뀌지 않는다. 바뀌는 것은 그 wrench-rate를 만드는 actuator 분배다.
2. 실제 구현에서는 actuator-rate 포화, 최종 위치/추력 clamp, 유한한 제어 주기, servo 동역학 때문에 `W_INV_f`가 최종 wrench 추종에도 영향을 줄 수 있다.
3. 수평인 HB에서 thrust 열 `J_f`는 `F_x`, `F_y`를 직접 만들 수 없다. 반면 beta bias가 있으면 thrust 차등이 즉시 `F_x`를 만든다.
4. tilt 상태의 thrust 차등은 `F_x`뿐 아니라 `M_y`도 함께 만든다. 따라서 정확한 pure-`F_x` wrench를 만족하려면 servo가 이 모멘트를 상쇄해야 하며, `W_INV_f`만 무한히 키워도 thrust 분담률은 무한히 증가하지 않는다.
5. MuJoCo HB plant는 thrust를 algebraic input으로 즉시 적용한다. 따라서 thrust command는 가상의 1차 지연 역모델이 아니라 `f_cmd = f + dt * f_dot`로 적분해야 한다.
6. 포화는 primary wrench-rate scale을 먼저 최대화하고, 남은 rate 여유에 nullspace motion을 넣는 lexicographic 방식으로 변경했다.

## HB wrench와 Jacobian 구조

Allocator 상태는 다음 순서다.

```text
q = [alpha1 alpha2 alpha3 alpha4 beta1 beta2 f1 f2 f3 f4]^T
```

각 rotor의 body-frame thrust 방향을 `e_i(alpha_i,beta_j)`, 위치를 `r_i`, 회전 방향 부호를 `s_i in {+1,-1}`, 반작용 모멘트 계수를 `zeta`라 하면,

```text
W(q) = sum_i f_i [r_i x e_i + s_i zeta e_i; e_i]
```

이고 wrench 순서는 다음과 같다.

```text
W = [Mx My Mz Fx Fy Fz]^T
```

따라서 `J=dW/dq`의 열은 다음 구조를 가진다.

```text
J_alpha_i = f_i * d([moment; force]_i) / d(alpha_i)
J_beta_j  = 해당 beta를 공유하는 두 rotor의 beta 미분 열의 합
J_f_i     = [r_i x e_i + s_i zeta e_i; e_i]
```

수평 hover에서 계산한 rank는 다음과 같다.

| 행렬 | rank |
|---|---:|
| `J_servo` (6x6) | 4 |
| `J_f` (6x4) | 4 |
| `J=[J_servo J_f]` (6x10) | 6 |

전체 `J`의 singular value는 다음과 같았다.

```text
[24.2785, 17.1675, 3.00598, 2.00000, 0.92699, 0.35000]
```

즉 servo와 thrust를 함께 쓰면 6-DoF wrench-rate가 locally controllable하지만, 수평 상태의 `J_f`에는 `F_x`, `F_y` 방향이 없다. 이때 `F_x`는 먼저 servo가 축을 꺾어야 생성된다.

## `W_INV_f`와 nullspace의 정확한 의미

코드의 weighted right inverse는 다음과 같다.

```text
J# = W_inv J^T (J W_inv J^T)^-1
```

`HB_W_INV`는 비용행렬 자체가 아니라 그 역행렬에 해당한다. 따라서 thrust 대각 원소 `W_INV_f`를 크게 하면 thrust-rate 좌표가 더 싸져 더 많이 사용된다.

Primary와 nullspace rate는 다음과 같다.

```text
q_dot_p = J# W_dot_des
q_dot_n = (I - J# J) q_dot_star
```

`J`가 full row rank이면,

```text
J J# = I
J (I - J# J) = 0
```

이므로 포화 전에는

```text
J (q_dot_p + q_dot_n) = W_dot_des
```

가 성립한다. 즉 beta bias를 nullspace에 넣는 것 자체는 현재 `q`에서 primary wrench-rate를 훼손하지 않는다.

다만 이것은 rate-level의 국소 관계다. 다음 항목까지 보장하는 식은 아니다.

- 유한한 한 step 뒤의 비선형 `W(q_k+1)=W_des`
- servo 실제 각도가 command를 즉시 따라감
- actuator-rate 포화 이후에도 scale 1 유지
- 최종 alpha/beta/thrust clamp 이후 exact wrench 유지

따라서 `W_INV_f`를 바꾼 뒤 wrench가 달라지는 현상은 nullspace 식의 오류가 아니라, 주로 분배가 바뀌면서 어떤 actuator가 먼저 포화되는지 달라지기 때문이다.

## Tilt 상태에서 thrust가 만드는 `F_x`와 필연적인 `M_y`

대칭 bias를 다음처럼 둔다.

```text
beta1 = -beta
beta2 = +beta
alpha_i = 0
```

그룹 1/4의 thrust를 `+delta_f`, 그룹 2/3의 thrust를 `-delta_f`만큼 바꾸면,

```text
Fx = 4 delta_f sin(beta)
My = 4 L delta_f cos(beta)
```

따라서

```text
My / Fx = L cot(beta)
```

이다. `L=0.175 m`, `beta=25 deg`이면,

```text
My / Fx = 0.375289 m
```

이다. 즉 thrust 차등으로 `F_x`를 늘리면 pitch moment가 자동으로 따라온다. pure `F_x` wrench를 유지하려면 servo가 반대 `M_y`를 만들어야 한다.

20 N/s의 pure-`F_x` wrench-rate에 대한 weighted inverse의 thrust 분담은 다음과 같다.

| beta | `W_INV_f` | thrust가 담당한 `F_x` 비율 |
|---:|---:|---:|
| 0 deg | 1~1,000,000 | 0.000% |
| 25 deg | 1 | 2.647% |
| 25 deg | 10 | 11.555% |
| 25 deg | 100 | 17.415% |
| 25 deg | 300 | 18.095% |
| 25 deg | 1,000 | 18.346% |
| 25 deg | 1,000,000 | 18.455% |

이 약 18.5%는 HB가 만들 수 있는 절대 물리 상한이 아니라, 현재의 exact-wrench 제약과 servo weight를 둔 weighted minimum-norm 해의 점근값이다. `W_INV_f=300` 이후의 증가는 매우 작으므로 기본값은 300으로 정했다. 더 높은 thrust 분담이 필요하면 이후에는 `W_INV_f`만 올리는 대신 beta-rate 비용 또는 별도의 thrust-priority 목적함수를 설계해야 한다.

## 추력 command 동역학 수정

HB MuJoCo plant는 받은 `f[i]`를 그 physics step의 rotor force에 직접 사용한다. 별도의 motor lag state가 없다.

기존 코드는 다음 가상 1차 동역학의 역을 사용했다.

```text
f_cmd = f + tau_f * f_dot
tau_f = 0.001 s
```

제어 주기 `dt=1/400=0.0025 s`에서 plant가 command를 즉시 적용하므로 실제 이산 rate는

```text
(f_cmd - f) / dt = (tau_f / dt) f_dot = 0.4 f_dot
```

밖에 되지 않았다. 이는 allocator가 계산한 thrust-rate의 40%만 실현하는 모델 불일치다. 현재 구현은 다음과 같다.

```text
f_cmd,k+1 = f_k + dt * f_dot_k
```

따라서 clamp가 없을 때 plant에 적용되는 discrete thrust-rate가 allocator의 `f_dot`와 정확히 일치한다.

Servo는 MuJoCo position actuator 동역학이 있으므로 기존 Eq. (14) 형태를 유지한다.

```text
servo_cmd = servo_measured + HB_Q_CMD_TAU_SERVO * servo_dot
```

## Primary-priority 포화

기존 구현은 primary와 nullspace rate를 더한 뒤 하나의 공통 scale `k_s`를 곱했다.

```text
q_dot = k_s (q_dot_p + q_dot_n)
```

이 경우 nullspace bias가 rate limit에 걸리면 primary wrench-rate도 같은 `k_s`만큼 불필요하게 감소한다.

현재 구현은 두 scale을 분리한다.

```text
q_dot = k_p q_dot_p + k_n q_dot_n
```

다음 제약 안에서 먼저 `k_p`를 최대화하고, 그 `k_p`에서 `k_n`을 최대화한다.

```text
maximize lexicographically (k_p, k_n)
subject to
  |k_p q_dot_p[i] + k_n q_dot_n[i]| <= q_dot_max[i]
  0 <= k_p <= 1
  0 <= k_n <= 1
```

고정된 `k_p`에서 각 actuator가 허용하는 `k_n` interval을 교차하고, feasible한 최대 `k_p`를 binary search한다. 이 방식은 nullspace rate가 primary rate를 상쇄해 주는 경우도 활용한다.

기본 rate limit은 다음처럼 완화했다.

```text
servo:  1.0 -> 1.5 rad/s
thrust: 100 -> 500 N/s
```

실제 임펄스 로그의 첫 1초를 같은 `q`, `W_des`로 재생한 결과는 다음과 같다.

| 경우 | 포화 방식/한계 | 최소 primary scale | 평균 primary scale | scale < 0.999 비율 |
|---|---|---:|---:|---:|
| tilt 0 | 기존 common, 1 rad/s·100 N/s | 0.5153 | 0.9448 | 27.68% |
| tilt 0 | 새 common, 1.5 rad/s·500 N/s | 0.7729 | 0.9881 | 9.23% |
| tilt 0 | 새 lexicographic | 0.7853 | 0.9896 | 8.48% |
| tilt 25 | 기존 common, 1 rad/s·100 N/s | 0.5448 | 0.9623 | 12.72% |
| tilt 25 | 새 common, 1.5 rad/s·500 N/s | 0.8173 | 0.9935 | 5.99% |
| tilt 25 | 새 lexicographic | 0.8174 | 0.9935 | 5.99% |

즉 rate limit 완화가 대부분의 wrench-rate 손실을 줄였고, lexicographic 처리도 nullspace 때문에 primary가 추가로 줄어드는 경우를 제거했다.

## 수치 검증

고정 seed로 무작위 operating state를 생성해 다음을 확인했다.

| 검증 | 표본 수 | 최대 오차 |
|---|---:|---:|
| analytic `J` 대 central finite difference | 1,000 | 상대오차 2.852×10^-10 |
| `J J# = I` | 10,000 | Frobenius 1.919×10^-13 |
| `J(I-J#J) = 0` | 10,000 | Frobenius 2.812×10^-13 |
| 새 포화 후 actuator-rate limit | 10,000 | 초과 5.507×10^-14 |
| `J q_dot = k_p W_dot_des` | 10,000 | 7.222×10^-14 |

같은 새 rate limit에서 기존 common scale과 비교했을 때 새 lexicographic 방식의 primary 보존율이 낮아진 경우는 10,000개 중 0개였다. 평균 개선량은 0.001293, 최대 개선량은 0.046931이었다.

별도로 analytic HB wrench를 MuJoCo의 rotor 위치, 축 방향 및 reaction torque 식과 1,000개 무작위 상태에서 비교한 최대 상대오차는 4.549×10^-16이었다. 따라서 이번 문제의 원인은 rotor geometry, sign 또는 frame 불일치가 아니다.

## 느린 servo plant 모델과 pre-tilt 채널

HB의 rotor thrust는 계속 algebraic input으로 즉시 적용한다. Servo만 plant 내부에서 다음 1차 모델을 거친다.

```text
tau_p * z_dot + z = u_servo
q_xml_ref = z
tau_p = 0.20 s
```

400 Hz physics step에서는 zero-order-hold 입력에 대한 정확한 이산해를 사용한다.

```text
a = exp(-dt / tau_p)
z[k+1] = a z[k] + (1-a) u_servo[k]
dt = 0.0025 s
```

따라서 수치적 Euler 근사 오차 없이 command에서 XML reference까지 정확한 1차 전달함수다. XML position actuator는 이 reference를 충분히 빠르게 추종하도록 다음처럼 설정했다.

| 축 | `kp` | `kv` |
|---|---:|---:|
| beta 1~4 | 600 | 9.0 |
| alpha 1~4 | 300 | 2.7 |

실제 6회 비행 전체에서 encoder와 LPF reference의 오차는 다음과 같았다.

| 조건 | RMS tracking error | max tracking error |
|---|---:|---:|
| tilt 0 | 0.0398 deg | 0.3135 deg |
| pre-tilt 25 | 0.0293 deg | 0.3134 deg |

즉 관측된 느린 응답은 XML servo gain 부족이 아니라 의도한 `tau_p=0.20 s` 1차 LPF가 지배한다.

Allocator의 Eq. (14) command inverse는 기존 `HB_Q_CMD_TAU_SERVO=0.05 s`를 유지했다.

```text
u_servo = q_measured + tau_a * q_dot
tau_a = 0.05 s
```

`tau_a=tau_p=0.20 s`로 맞추면 이상적인 선형 구간에서 `(tau_a s+1)/(tau_p s+1)=1`이 되어, 비교를 위해 넣은 느린 plant를 allocator가 거의 상쇄한다. 실제 첫 iteration에서도 이 현상이 확인됐기 때문에 plant는 정확한 0.20 s 1차 모델로 두되 allocator는 이를 완전히 알지 못하는 0.05 s 설정으로 되돌렸다. 이는 servo가 예상보다 느린 경우에도 pre-tilt thrust channel이 유리한지 보는 의도적 stress condition이다.

### 왜 pre-tilt에서 thrust가 즉시 병진력에 관여하는가

작은 actuator 변화에 대해 x축 force 증분은 다음처럼 나뉜다.

```text
delta Fx = J_q,Fx delta q + J_f,Fx delta f
J_f,Fx[i] = -sin(beta_i) cos(alpha_i)
```

수평 상태 `beta_i=0`에서는 `J_f,Fx=0`이므로 thrust 크기를 즉시 바꿔도 x축 힘은 생기지 않는다. 먼저 servo가 축을 꺾어야 하고,

```text
delta q(s) = 1 / (tau_p s + 1) * delta u_servo(s)
```

의 지연을 피할 수 없다.

반면 `beta_i != 0`인 pre-tilt 상태에서는 `J_f,Fx != 0`이다. Rotor thrust가 plant에 algebraic input으로 들어가므로,

```text
delta Fx_thrust = sum_i -sin(beta_i) cos(alpha_i) delta f_i
```

가 같은 physics step에 생성된다. 즉 비틸트의 x축 servo 경로는 1차 지연을 갖지만, pre-tilt의 thrust-modulation 경로에는 direct feedthrough가 있다. 이것이 이번 비교에서 의도한 강건성 차이다. 다만 앞에서 유도한 `My/Fx=L cot(beta)` 결합 때문에 exact 6-DoF wrench를 유지하려면 servo 보상이 여전히 필요하다.

## MuJoCo 실제 임펄스 비행 비교

### 조건

- vehicle: HB A4B2
- controller/physics: 400 Hz
- plant servo LPF: `tau_p=0.20 s`
- allocator servo command constant: `tau_a=0.05 s`
- `W_INV_f=300`
- servo-rate limit: 1.5 rad/s
- thrust-rate limit: 500 N/s
- 외란: controller world `+x`, 10 N, 0.1 s
- noise/random disturbance: off
- position hover: `x=y=0`, `z=1 m`
- 각 조건 실제 임펄스 3회
- tilt 없음: 실제 `abs(beta)=0.0019 +/- 0.0016 deg`
- tilt 있음: 초기 관절과 nullspace target을 모두 `[-25,+25] deg`로 설정, 임펄스 직전 실제 `abs(beta)=24.0299 +/- 0.2459 deg`

두 조건에서 측정된 외란 peak는 모두 `10.0 +/- 0.0 N`, 적분 impulse는 모두 `1.0167 +/- 0.0144 N s`였다. 따라서 아래 차이는 호버링 관찰이 아니라 동일한 실제 plant impulse 이후의 이동 응답 차이다.

위치 RMS는 임펄스 이후 8초 구간의

```text
RMS_x = sqrt(1/T * integral_0^T x_error(t)^2 dt)
```

로 계산했다. Settling time은 `abs(x error)<=2 cm`와 `abs(vx)<=5 cm/s`를 이후 계속 만족하는 최초 시각이다.

### 위치 응답 결과

| 지표, mean +/- sample std | tilt 없음 | pre-tilt 25 | 변화 |
|---|---:|---:|---:|
| x RMS, 0~8 s | 0.030847 +/- 0.000737 m | 0.024061 +/- 0.000484 m | 22.00% 감소 |
| peak x 변위 | 0.082220 +/- 0.001401 m | 0.077831 +/- 0.001372 m | 5.34% 감소 |
| 반대방향 overshoot | 0.065205 +/- 0.001879 m | 0.051497 +/- 0.001025 m | 21.02% 감소 |
| integral(0..8) abs(x) dt | 0.188614 +/- 0.004805 m s | 0.124733 +/- 0.002898 m s | 33.87% 감소 |
| settling time | 6.9108 +/- 0.0419 s | 4.2492 +/- 0.0275 s | 2.6617 s, 38.51% 단축 |
| peak abs(vx) | 0.285879 +/- 0.003878 m/s | 0.285305 +/- 0.003910 m/s | 0.20% 감소 |
| peak abs(pitch error) | 0.028895 +/- 0.000012 deg | 0.112498 +/- 0.000545 deg | 0.0836 deg 증가 |

요청한 두 핵심 지표인 x RMS와 peak가 모두 감소했다. Peak 속도가 거의 같은 것은 10 N, 0.1 s의 동일한 외란이 feedback가 충분히 반응하기 전에 같은 운동량을 주기 때문이다. 차이는 이후 제동과 복귀에서 크게 나타났다. Pitch error는 증가했지만 절대 peak는 0.113 deg 이하였고, 이는 thrust 차등이 `My`를 함께 만드는 구조와 일치한다.

### thrust modulation과 servo 방향변화 분리

임펄스 전 rotor 축을 `e_x_i_0`, 추력을 `f_i_0`라 두고 수평력 변화를 다음처럼 분리했다.

```text
Delta Fx_thrust = sum_i (f_i - f_i,0) e_x_i_0
Delta Fx_servo  = Delta Fx_total - Delta Fx_thrust
```

| 지표, mean +/- sample std | tilt 없음 | pre-tilt 25 |
|---|---:|---:|
| 첫 0.15 s peak abs(Delta Fx_thrust) | 약 0 N | 0.16078 +/- 0.00247 N |
| 첫 1 s peak abs(Delta Fx_thrust) | 약 0 N | 0.66348 +/- 0.00429 N |
| 첫 1 s peak abs(Delta Fx_servo) | 3.71413 +/- 0.05373 N | 3.11514 +/- 0.05672 N |
| 첫 1 s peak sum abs(Delta f_i) | 0.17328 +/- 0.00452 N | 1.62937 +/- 0.01140 N |
| peak force command 시 thrust 분담 | 약 0% | 9.664 +/- 0.213% |

Pre-tilt에서는 첫 1초 rotor thrust activity가 9.40배로 증가하고 servo 방향변화가 담당한 peak x-force는 16.13% 감소했다. 따라서 성능 향상이 servo를 더 빨리 움직여서 생긴 것이 아니라, 이미 기울어진 rotor의 즉시 반응하는 thrust 크기 채널을 더 사용한 결과라는 해석과 데이터가 일치한다.

## Beta 30 deg 상한 확인

향후 alpha도 최대 30 deg까지 pre-tilt한다는 제약을 고려해 beta만 25 deg에서 30 deg로 늘렸을 때의 의미를 별도로 확인했다. 이 시험에서는 alpha target은 계속 0 deg로 유지해 beta 효과만 분리했다.

고정 hover state에서 `W_INV_f=300`인 weighted right inverse를 계산한 결과는 다음과 같다.

| beta | rotor당 hover thrust | `L cot(beta)` | pure-Fx rate의 thrust 분담 | 요구 max servo rate |
|---:|---:|---:|---:|---:|
| 25 deg | 9.471 N | 0.3753 m | 18.095% | 0.477 rad/s |
| 30 deg | 9.912 N | 0.3031 m | 25.256% | 0.435 rad/s |

즉 5 deg 증가로 x방향 thrust 축 성분은 18.3% 커지고, thrust가 동반하는 pitch moment 비율은 19.2% 작아진다. 두 효과가 함께 작용해 이론적 thrust 분담은 7.161%p 증가한다. Rotor당 hover thrust도 20 N limit의 절반 정도라 thrust 여유는 충분하다.

동일한 `tau_p=0.20 s`, controller gain, allocator parameter와 10 N, 0.1 s 실제 impulse를 사용해 30 deg 조건도 3회 비행했다. 임펄스 직전 실제 `abs(beta)=28.3564 +/- 0.4548 deg`였다.

| 지표, mean +/- sample std | pre-tilt 25 | pre-tilt 30 | 25 -> 30 변화 |
|---|---:|---:|---:|
| x RMS, 0~8 s | 0.024061 +/- 0.000484 m | 0.022388 +/- 0.000519 m | 6.95% 감소 |
| peak x 변위 | 0.077831 +/- 0.001372 m | 0.076077 +/- 0.001415 m | 2.25% 감소 |
| 반대방향 overshoot | 0.051497 +/- 0.001025 m | 0.047375 +/- 0.001160 m | 8.00% 감소 |
| integral(0..8) abs(x) dt | 0.124733 +/- 0.002898 m s | 0.108890 +/- 0.003227 m s | 12.70% 감소 |
| settling time | 4.2492 +/- 0.0275 s | 3.4725 +/- 0.0198 s | 18.28% 단축 |
| peak force command 시 thrust 분담 | 9.664 +/- 0.213% | 12.986 +/- 0.418% | 3.322%p 증가 |
| 첫 0.15 s peak abs(Delta Fx_thrust) | 0.16078 +/- 0.00247 N | 0.23135 +/- 0.00258 N | 43.90% 증가 |
| 첫 1 s peak abs(Delta Fx_servo) | 3.11514 +/- 0.05672 N | 2.89101 +/- 0.06580 N | 7.20% 감소 |
| peak abs(pitch error) | 0.11250 +/- 0.00055 deg | 0.13285 +/- 0.00130 deg | 0.02035 deg 증가 |

비틸트와 30 deg를 비교하면 x RMS는 27.42%, peak는 7.47%, IAE는 42.27%, settling time은 49.75% 감소했다. 따라서 beta 30 deg는 물리적으로 의미가 있고 현재의 30 deg 제한에서는 25 deg보다 우수하다.

다만 25 deg 대비 peak 추가 개선은 2.25%이므로 각도 5 deg만으로 압도적인 peak 차이를 만들지는 못한다. 더 큰 차이는 x position/wrench gain이 빠른 thrust channel을 더 요구하도록 조정하거나, 느린 servo stress를 강화하는 방법과 결합해야 한다. Servo를 더 느리게 하는 것은 비틸트 성능을 의도적으로 악화시키는 효과도 있으므로 절대 성능 개선과 강건성 stress test를 구분해 해석해야 한다.

향후 alpha와 beta를 모두 30 deg로 둘 경우 thrust 축의 수직 성분은 `cos(alpha) cos(beta)=0.75`가 되고, beta가 만드는 x축 성분에는 `cos(alpha)=0.866`이 곱해진다. 따라서 이번 alpha=0 시험의 절대 x-force 크기를 그대로 사용할 수 없으며, alpha/beta 동시 bias에 대한 새로운 `J`와 실제 impulse 검증이 필요하다. x축 외란이 주 관심이면 beta를 30 deg, alpha를 그보다 작게 두는 불균등 pre-tilt도 수학적으로 타당한 후보이다.

## 최종 stress tuning: tau_p=0.35 s, K_J(Fx,Fy)=40, beta target 44 deg

앞의 `tau_p=0.20 s`, beta 25/30 deg 결과는 1차 검증 단계의 기록이다. Servo 지연 차이를 더 명확히 드러내고 빠른 thrust 채널을 적극적으로 쓰기 위해 다음 순서로 추가 실험했다.

- plant servo LPF: `tau_p=0.20 -> 0.35 s`
- allocator servo command constant: `tau_a=0.05 s` 유지
- HB 수평 force gain만 `K_J(Fx,Fy)=20 -> 30 -> 40 -> 50` 순서로 증가
- beta target: 30, 40, 44, 45 deg 비교
- torque축과 `Fz`의 `K_J`는 변경하지 않음
- Hexa의 `K_J`도 기존 `[60,60,60,20,20,20]`으로 분리해 변경하지 않음

### 느린 servo에서 K_J를 올릴 때 pre-tilt가 필요한 이유

구현의 wrench-rate 명령은 다음과 같다.

```text
W_dot_des = K_J (W_des - W(q))
q_dot_primary = J# W_dot_des
```

따라서 `K_J(Fx,Fy)`를 20에서 40으로 올리면 같은 수평 wrench 오차에 대해 요구 actuator rate가 두 배가 된다. Full-row-rank이고 실제 actuator가 계산된 `q_dot`를 그대로 만들면 `J q_dot_primary=W_dot_des`가 성립한다. 그러나 현재 stress model에서는 servo command inverse와 plant가 각각

```text
u_servo = q_measured + tau_a q_dot_cmd,   tau_a = 0.05 s
q_actual / u_servo = 1 / (tau_p s + 1),  tau_p = 0.35 s
```

이므로 servo 경로의 command-to-angle 비는

```text
(tau_a s + 1) / (tau_p s + 1)
```

이고 고주파 이득은 `tau_a/tau_p=0.1429`뿐이다. 즉 allocator가 계산한 빠른 servo rate를 실제 plant가 같은 속도로 만들지 못하며, 이때는 `J q_dot_actual=W_dot_des`가 더 이상 성립하지 않는다.

수평 상태에서는

```text
J_f,Fx = [0, 0, 0, 0]
```

이므로 수평 force 보정의 100%가 이 느린 servo 경로를 거친다. 반면 대칭 beta 44 deg, alpha 0 상태에서는 rotor 순서에 대해

```text
J_f,Fx = [+sin(44 deg), -sin(44 deg), -sin(44 deg), +sin(44 deg)]
       = [+0.69466, -0.69466, -0.69466, +0.69466]
```

가 되어 thrust 변화가 같은 physics step에 `Fx`를 만든다.

Beta 44 deg hover에서 계산한 전체 `J`는 여전히 rank 6이고 최소 singular value는 0.18230이다. `W_INV_f=300`인 weighted inverse에 pure-`Fx` wrench-rate 20 N/s를 넣으면 다음 분배가 나온다.

| 항목 | beta 44 deg |
|---|---:|
| thrust 경로의 `Fx` 분담 | 48.3047% |
| servo 경로의 `Fx` 분담 | 51.6953% |
| max servo rate | 0.3011 rad/s |
| max thrust rate | 3.4769 N/s |
| thrust가 동반하는 `My/Fx=L cot(beta)` | 0.18122 m |

즉 pre-tilt가 nullspace 등식을 바꾸는 것이 아니다. 동일한 exact-wrench weighted inverse 안에서 실제 plant가 빠르게 실현할 수 있는 thrust 열이 `Fx` 행에 생기고, 느린 servo에만 의존하던 비율이 줄어드는 것이다. 포화 전 수학적 nullspace 직교성은 그대로이며, 차이는 command rate와 실제 actuator rate 사이의 동역학에서 발생한다.

### K_J 증가 실험

Beta target 45 deg에서 각 gain을 독립적으로 한 번씩 먼저 시험했다. 모두 동일한 10 N, 0.1 s x-impulse와 `tau_p=0.35 s` 조건이다.

| `K_J(Fx,Fy)` | x RMS | peak x | IAE | settling | peak pitch |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.035659 m | 0.083757 m | 0.233631 m s | 7.9600 s | 0.2659 deg |
| 30 | 0.021678 m | 0.079944 m | 0.090407 m s | 2.6525 s | 0.4147 deg |
| 40 | 0.020007 m | 0.079210 m | 0.070854 m s | 1.8925 s | 0.5662 deg |
| 50 | 0.019958 m | 0.079847 m | 0.067703 m s | 1.8675 s | 0.7361 deg |

40에서 50으로 올렸을 때 RMS 개선은 0.25%, settling 개선은 1.32%에 그쳤다. 반면 peak x는 0.80% 증가했고 peak pitch는 30.0% 증가했다. 따라서 안정한 범위에서 이득을 올리되 한계효용과 자세 coupling을 고려해 최종값은 40으로 정했다.

Beta target 30 deg, `K_J=20`은 같은 느린 servo 조건에서 임펄스 후 pitch가 89.97 deg까지 증가하고 settling하지 못했다. Beta를 키우는 것이 단순한 hover 자세 변경이 아니라 실제 안정 여유를 회복하는 데 의미가 있음을 확인했다. 같은 `K_J=40`에서 beta 40/44/45 deg의 1회 결과는 다음과 같다.

| beta target | x RMS | peak x | settling | 실제 hover peak thrust | impulse 포함 peak thrust |
|---:|---:|---:|---:|---:|---:|
| 40 deg | 0.023928 m | 0.083148 m | 2.6650 s | 54.07% | 57.76% |
| 44 deg | 0.021052 m | 0.080558 m | 1.9375 s | 56.34% | 60.43% |
| 45 deg | 0.020007 m | 0.079210 m | 1.8925 s | 57.46% | 61.56% |

45 deg가 절대 성능은 가장 좋지만, 향후 alpha도 30 deg까지 기울일 때의 hover 추력 여유를 함께 고려해 44 deg를 최종값으로 선택했다.

### 최종 무틸트 대 pre-tilt 반복 비교

최종 조건은 `tau_p=0.35 s`, `K_J(Fx,Fy)=40`, `W_INV_f=300`, servo/thrust rate limit 1.5 rad/s와 500 N/s이다. 무틸트와 beta target 44 deg를 각각 3회 비행했다. 실제 임펄스는 모든 run에서 10.0 N, 0.1 s, 1.0000 N s였다.

| 지표, mean +/- sample std | beta 0 | beta target 44 deg |
|---|---:|---:|
| 임펄스 직전 실제 abs(beta) | 약 0 deg | 40.5138 +/- 0.1623 deg |
| x RMS, 0~8 s | 0.758729 +/- 0.270293 m | 0.020822 +/- 0.000206 m |
| peak x | 1.235883 +/- 0.223540 m | 0.080333 +/- 0.000207 m |
| IAE | 5.007932 +/- 1.728035 m s | 0.076467 +/- 0.001462 m s |
| settling time | 3회 모두 실패 | 1.9283 +/- 0.0080 s |
| peak abs(vx) | 1.944770 +/- 0.404279 m/s | 0.275404 +/- 0.000103 m/s |
| peak abs(pitch) | 89.6975 +/- 0.2501 deg | 0.5551 +/- 0.0027 deg |
| peak force command | 50.0000 N | 5.1450 +/- 0.0088 N |
| 첫 0.15 s thrust `Fx` | 약 0 N | 0.5382 +/- 0.0041 N |
| 첫 1 s thrust `Fx` | 약 0 N | 1.9475 +/- 0.0126 N |
| 첫 1 s servo-direction `Fx` | 3.6808 +/- 0.0003 N | 2.2414 +/- 0.0096 N |

Pre-tilt의 RMS는 97.26%, peak x는 93.50%, IAE는 98.47% 감소했다. 다만 이 상대값은 두 안정한 선형 응답의 작은 차이가 아니라, 무틸트가 3회 모두 자세를 잃은 것과 pre-tilt가 3회 모두 안정한 것의 차이다. 따라서 최종 결론은 “몇 % 성능 향상”보다 “느린 servo와 높은 수평 wrench gain에서 pre-tilt가 비행 안정성을 유지했다”가 더 정확하다.

Target과 실제 beta가 다른 것은 beta bias가 hard equality constraint가 아니라

```text
(I - J#J) q_dot_star
```

로 투영된 nullspace 목표이기 때문이다. Steady state에서 `q=q_ref`가 아니더라도 `q_dot_star`의 남은 성분이 `J`의 row-space에 있으면 nullspace projection이 0이 될 수 있다. 이것은 primary wrench를 버린 결과가 아니다.

### Hover thrust 한계

최종 beta 44 deg 반복 로그에서 측정한 rotor 사용률은 다음과 같다.

| 항목 | 측정값 |
|---|---:|
| hover mean rotor thrust | 11.2907 +/- 0.0273 N, 56.45% |
| hover peak rotor thrust | 11.2969 +/- 0.0277 N, 56.48% |
| impulse 포함 peak rotor thrust | 12.1045 +/- 0.0212 N, 60.52% |

정확히 beta 44 deg, alpha 0인 정적 기하에서는 rotor당 hover thrust가 11.9328 N, 즉 20 N limit의 59.66%다. 향후 alpha=30 deg까지 동시에 pre-tilt한다고 가정하면

```text
f_hover = mg / (4 cos(alpha) cos(beta))
        = 13.7788 N = 68.89% of 20 N
```

이다. 따라서 현재 beta 44 deg는 alpha 30 deg 계획까지 포함한 이상적 정적 계산에서도 70% 아래다. 이 68.89%는 alpha/beta 동시 틸트의 실제 폐루프 비행 검증값이 아니라 설계 상한 계산이므로, alpha를 구현한 뒤에는 같은 impulse와 hover 로그를 다시 수행해야 한다.

## 반복 실험 및 비교 로그 도구

`plant` 패키지에 두 실행 파일을 추가했다.

- `hb_impulse_logger`: 실제 `/test/impulse_trigger`를 publish하고 state/input/wrench 원시 CSV와 metadata JSON을 저장한다.
- `hb_impulse_compare`: 지정한 두 `label_run*` 집합을 시간 정렬해 run별/평균 지표, hover/peak rotor 사용률, JSON, CSV, mean +/- std overlay PNG를 만든다.

초기 0.20 s 실험은 `data_logs/hb_impulse_tau020_beta25/`에 있고, 최종 0.35 s gain/beta sweep과 3회 비교는 `data_logs/hb_impulse_tau035_kj_beta_sweep/`에 있다. 두 디렉터리는 생성 데이터이므로 Git ignore 대상이지만 workspace에는 그대로 유지된다.

한 회 기록 예시는 다음과 같다.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run plant hb_impulse_logger \
  --output-prefix data_logs/hb_impulse_tau035_kj_beta_sweep/beta44_kj40_run4 \
  --label beta44_kj40 --warmup 10 --post 8
```

두 조건의 반복 로그 비교는 다음과 같다.

```bash
ros2 run plant hb_impulse_compare \
  --data-dir data_logs/hb_impulse_tau035_kj_beta_sweep \
  --untilted-label beta0_kj40 \
  --tilted-label beta44_kj40 \
  --output-prefix comparison_final_tau035_kj40_beta0_vs44 \
  --window 8
```

생성 파일은 다음과 같다.

- `comparison_final_tau035_kj40_beta0_vs44.png`: 최종 beta 0 대 beta 44 overlay
- `comparison_final_tau035_kj40_beta0_vs44_summary.csv`: mean, sample standard deviation, 상대 변화
- `comparison_final_tau035_kj40_beta0_vs44_runs.csv`: 6개 개별 run 지표
- `comparison_final_tau035_kj40_beta0_vs44.json`: 위 결과 전체와 각 run metadata
- 비교 JSON/CSV에는 hover 및 impulse 구간의 rotor thrust와 20 N 대비 사용률도 포함된다.

## 실행 파라미터

현재 기본 실행은 최종 검증값을 사용한다.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch multirotor_cmd run.py vehicle:=hummingbird mode:=position_cmd
```

위 명령의 HB 기본값은 다음과 같다.

```text
hb_beta_bias_deg=44.0
hb_beta_initial_deg=44.0
hb_servo_tau_sec=0.35
hb_kj_force_xy=40.0
hb_w_inv_f=300.0
hb_servo_rate_max=1.5
hb_thrust_rate_max=500.0
```

공정한 무틸트 비교는 beta 두 인자만 0으로 명시하고 나머지는 같게 둔다.

```bash
ros2 launch multirotor_cmd run.py \
  vehicle:=hummingbird mode:=position_cmd \
  hb_beta_bias_deg:=0.0 hb_beta_initial_deg:=0.0 \
  hb_servo_tau_sec:=0.35 hb_kj_force_xy:=40.0 \
  hb_w_inv_f:=300.0 \
  hb_servo_rate_max:=1.5 hb_thrust_rate_max:=500.0
```

`hb_beta_initial_deg`는 MuJoCo 관절과 plant LPF state를 실험 시작부터 target tilt에 맞춰, 느린 nullspace 수렴 대기시간이 결과에 섞이지 않게 한다.

## 현재 해석의 범위

이번 결과는 noise와 random disturbance를 끈 deterministic MuJoCo 실제 impulse 실험이다. 동일한 controller/allocator gain, rate limit 및 servo LPF에서 pre-tilt 여부만 바꿨다. 최종 stress 조건에서는 무틸트가 3회 모두 자세를 잃고 beta 44 deg가 3회 모두 안정했으므로, 현재 모델에서는 “pre-tilt가 즉시 thrust 병진 채널을 열어 느린 servo와 높은 수평 wrench gain에 대한 안정 여유를 만든다”는 가설이 강하게 지지된다.

질량/관성 오차, motor lag, thrust calibration 오차와 stochastic disturbance를 함께 바꾼 Monte Carlo 강건성은 별도의 다음 실험 범위다.
