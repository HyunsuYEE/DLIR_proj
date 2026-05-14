# DPM-Solver 분석 및 DIAMOND 통합 계획

## 1. suggestion.md 기준 목표 이해

`docs/suggestion.md`의 목표는 DIAMOND actor-critic training loop 안에서 diffusion world model inference가 차지하는 시간을 줄이되, 최종 return을 보존하는 것이다. DPM-Solver는 이 목표에서 TeaCache와 다른 역할을 한다.

- TeaCache는 같은 denoising trajectory 안에서 일부 model forward를 cache로 생략하는 방식이다.
- DPM-Solver는 sampler 자체를 더 높은 차수의 ODE solver로 바꿔 같은 품질을 더 적은 function evaluation으로 얻으려는 방식이다.

따라서 DPM-Solver는 `suggestion.md`의 baseline 중 “정적 faster sampler” 또는 aggressive mode 후보로 적합하다. 다만 DIAMOND의 현재 sampler는 EDM/Karras sigma parameterization에 가까운 Euler/Heun sampler이고, 로컬 `dpm-solver` repo의 공식 구현은 VP diffusion ODE 기준이다. 이 차이를 해결하지 않으면 단순 copy-paste는 위험하다.

## 2. 논문 핵심 정리

DPM-Solver 논문은 diffusion sampling을 reverse process simulation이 아니라 diffusion ODE를 푸는 문제로 본다. 핵심은 diffusion ODE 해의 linear part를 analytic하게 처리하고, neural network 부분만 exponential weighted integral로 근사하는 전용 high-order solver를 만든 점이다.

논문에서 중요한 내용:

- 기존 diffusion model은 수백~수천 번의 neural network evaluation이 필요하다.
- DPM-Solver는 training-free sampler이다. diffusion model을 재학습하지 않는다.
- discrete-time과 continuous-time DPM 모두 지원하는 것을 목표로 한다.
- 약 10~20 function evaluations에서 좋은 sample quality를 얻는다고 보고한다.
- 1차 update는 DDIM과 연결되고, 2차/3차 solver는 neural network output의 변화량을 더 높은 차수로 근사한다.

DPM-Solver++는 guided sampling 안정성을 개선한 후속 방법이다.

- DPM-Solver++는 data prediction model 관점에서 ODE를 풀며, pixel-space guided sampling에서는 dynamic thresholding도 지원한다.
- multistep DPM-Solver++는 큰 guidance scale에서 생기는 instability를 줄이기 위해 effective step size를 줄이는 방향으로 설계됐다.

DIAMOND는 classifier-free guidance가 없고 action-conditioned world model이다. 따라서 DPM-Solver++의 guided-sampling 이점보다는 “data prediction model 기반 high-order/multistep solver”라는 점이 더 관련 있다.

참고 자료:

- DPM-Solver 논문: https://arxiv.org/abs/2206.00927
- DPM-Solver++ 논문: https://arxiv.org/abs/2211.01095
- DPM-Solver repo: https://github.com/LuChengTHU/dpm-solver
- DIAMOND 논문: https://arxiv.org/abs/2405.12399

## 3. 로컬 DPM-Solver 프로젝트 코드 분석

로컬 경로는 `/workspace/dpm-solver`이다. 핵심 구현은 거의 `dpm_solver_pytorch.py` 하나에 들어 있다.

중요한 파일:

- `dpm-solver/README.md`
- `dpm-solver/dpm_solver_pytorch.py`
- `dpm-solver/dpm_solver_jax.py`
- `dpm-solver/examples/*`

### 3.1 핵심 메커니즘 1: NoiseScheduleVP

`dpm_solver_pytorch.py`의 `NoiseScheduleVP`는 VP diffusion의 alpha, sigma, lambda를 계산한다.

주요 개념:

```python
lambda_t = log(alpha_t) - log(sigma_t)
```

지원 schedule:

- `schedule="discrete"`: betas 또는 alphas_cumprod를 받아 discrete diffusion model을 continuous time으로 interpolation한다.
- `schedule="linear"`: continuous VPSDE의 linear beta schedule을 사용한다.

중요한 점은 이 구현이 VP schedule을 중심으로 설계되었다는 것이다. DIAMOND의 sampler는 `sigma_min`, `sigma_max`, `rho`로 Karras sigma schedule을 만들고, `D(x, sigma)` 형태의 denoised data prediction을 사용한다. 즉 `NoiseScheduleVP`를 그대로 쓰려면 DIAMOND의 sigma state와 VP state 사이의 mapping을 명확히 정의해야 한다.

### 3.2 핵심 메커니즘 2: model_wrapper

`model_wrapper()`는 여러 parameterization을 noise prediction function으로 변환한다.

지원 model type:

- `noise`
- `x_start`
- `v`
- `score`

DIAMOND의 `Denoiser.denoise()`는 noisy frame과 sigma를 받아 quantized denoised frame, 즉 data prediction `x0`에 가까운 값을 반환한다. 그러므로 DPM-Solver 관점에서는 `model_type="x_start"` 또는 DPM-Solver++의 data prediction 경로가 가장 자연스럽다.

하지만 DIAMOND의 `Denoiser.denoise()`는 내부적으로 `wrap_model_output()`에서 clamp/quantize를 수행한다. Solver가 중간 update에서 smooth output을 기대할 때 quantization이 오차를 키울 수 있으므로, DPM-Solver 실험에서는 두 경로를 비교해야 한다.

- 기존 `denoise()` 사용: DIAMOND 원래 sampling behavior와 일치
- `compute_model_output()` + continuous `x0` 사용: solver 수치 안정성에 유리할 수 있음

### 3.3 핵심 메커니즘 3: DPM_Solver class

`DPM_Solver`는 solver 설정과 update formula를 담는다.

중요 설정:

- `algorithm_type="dpmsolver"` 또는 `"dpmsolver++"`
- `correcting_x0_fn`
- `correcting_xt_fn`
- `dynamic_thresholding_ratio`
- `thresholding_max_val`

DIAMOND는 pixel-space model이고 output range가 `[-1, 1]`이다. DPM-Solver++의 dynamic thresholding은 pixel-space model에서 도움이 될 수 있지만, DIAMOND는 이미 `wrap_model_output()`에서 clamp와 byte quantization을 한다. 두 보정이 겹치면 transition이 과도하게 clipping될 수 있으므로 처음에는 dynamic thresholding을 끄는 것이 안전하다.

### 3.4 핵심 메커니즘 4: update 함수

공식 구현은 다음 update들을 포함한다.

- `dpm_solver_first_update()`: 1차 update. DDIM과 동등한 역할.
- `singlestep_dpm_solver_second_update()`: 같은 interval 안에서 중간 time `s1`을 평가하는 2차 single-step.
- `singlestep_dpm_solver_third_update()`: 3차 single-step.
- `multistep_dpm_solver_second_update()`: 이전 model output들을 이용하는 2차 multistep.
- `multistep_dpm_solver_third_update()`: 이전 model output들을 이용하는 3차 multistep.

DIAMOND integration에서 가장 먼저 실험할 후보는 2차 multistep이다.

이유:

- DPM-Solver++ README도 guided sampling에서 2차 multistep을 추천한다.
- 3차 solver는 작은 step 수에서 instability가 생길 수 있다.
- DIAMOND world model은 RL target에 직접 영향을 주므로, aggressive한 3차보다 2차가 더 보수적인 시작점이다.

### 3.5 핵심 메커니즘 5: sample()

`sample()`은 다음 method를 지원한다.

- `multistep`
- `singlestep`
- `singlestep_fixed`
- `adaptive`

DIAMOND에는 `adaptive`를 바로 넣기 어렵다. actor-critic training loop 안에서는 batch별 function evaluation 수가 달라지면 latency variance와 compile/cache behavior가 복잡해진다. 먼저 fixed NFE sampler를 구현하고, PRG가 batch 단위로 mode를 선택하게 하는 것이 낫다.

## 4. DIAMOND 쪽 현재 sampler와의 차이

현재 DIAMOND sampler는 `src/models/diffusion/diffusion_sampler.py`에 있다.

현재 경로:

```python
sigmas = build_sigmas(num_steps_denoising, sigma_min, sigma_max, rho)
x = torch.randn(...)
for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
    denoised = denoiser.denoise(x, sigma, prev_obs, prev_act)
    d = (x - denoised) / sigma
    x = x + d * (next_sigma - sigma)
```

`order == 1`이면 Euler이고, `order == 2`이면 Heun이다. 즉 이미 sigma-domain ODE solver 구조를 갖고 있다.

DPM-Solver 공식 구현과 다른 점:

- 공식 구현은 VP `alpha_t`, `sigma_t`, `lambda_t`를 기준으로 한다.
- DIAMOND는 Karras sigma schedule과 EDM-style denoiser preconditioning을 쓴다.
- 공식 구현은 model function이 noise prediction을 반환한다고 가정한 뒤 내부에서 data prediction으로 바꿀 수 있다.
- DIAMOND denoiser는 이미 data prediction `D(x, sigma)`를 직접 제공한다.
- DIAMOND는 action-conditioned autoregressive world model이므로, sampler error가 visual quality뿐 아니라 reward/end prediction과 value target에 영향을 준다.

## 5. DIAMOND 통합 계획

### 5.1 1단계: sampler interface 분리

먼저 기존 `DiffusionSampler`를 직접 복잡하게 만들지 말고 sampler backend를 분리한다.

새 config 후보:

```yaml
world_model_env:
  diffusion_sampler:
    type: euler_heun
    num_steps_denoising: 3
    sigma_min: 2e-3
    sigma_max: 5.0
    rho: 7
    order: 1
    dpm_solver:
      enabled: false
      algorithm_type: dpmsolver++
      method: multistep
      order: 2
      steps: 3
      schedule_mapping: edm_to_vp
      dynamic_thresholding: false
```

구현 위치:

- `src/models/diffusion/diffusion_sampler.py`
- 필요하면 `src/models/diffusion/dpm_solver_sampler.py` 새 파일 추가

목표:

- 기존 Euler/Heun baseline을 완전히 유지한다.
- config만 바꿔 DPM-Solver backend를 켤 수 있게 한다.
- timing metric은 기존 `WorldModelEnv.pop_timing_stats()`가 그대로 잡도록 한다.

### 5.2 2단계: DPM-Solver를 직접 copy하지 말고 필요한 부분만 이식

공식 `dpm_solver_pytorch.py`는 크고 VP 중심이다. DIAMOND에는 다음 부분만 직접 가져오는 것이 적합하다.

- timestep/order scheduler
- multistep previous model output 관리
- lower_order_final 정책
- 1차/2차 update 구조

반면 `NoiseScheduleVP`와 `model_wrapper`는 그대로 쓰기보다 DIAMOND sigma-domain에 맞게 adapter를 만드는 편이 안전하다.

### 5.3 3단계: 두 가지 mapping 실험

#### A안: VP adapter 방식

DIAMOND의 raw EDM state를 VP state로 바꿔 공식 DPM-Solver에 맞춘다.

가능한 mapping:

```text
alpha(sigma) = 1 / sqrt(1 + sigma^2)
sigma_vp(sigma) = sigma / sqrt(1 + sigma^2)
x_vp = alpha * x_edm
x_edm = x_vp / alpha
```

DIAMOND denoiser call:

```text
x0 = Denoiser.denoise(x_edm, sigma, obs, act)
```

이 방식은 공식 DPM-Solver 수식에 가까워지는 장점이 있지만, state scaling과 initial noise scaling이 맞는지 반드시 검증해야 한다.

#### B안: EDM sigma-domain DPM-Solver++ 방식

DIAMOND의 현재 ODE를 유지한다.

```text
dx/dsigma = (x - D(x, sigma)) / sigma
```

여기에 DPM-Solver++류 multistep idea를 sigma 또는 log-sigma domain으로 옮긴다. 이 방식은 공식 repo를 그대로 쓰지는 않지만, DIAMOND의 현재 sampler와 가장 호환성이 좋다.

초기 구현은 다음 순서가 현실적이다.

1. 현재 Euler/Heun을 그대로 둔다.
2. `model_prev_list`와 `sigma_prev_list`를 저장한다.
3. 2차 Adams-Bashforth 형태로 derivative를 extrapolate한다.
4. 같은 NFE에서 Heun 대비 quality/return을 비교한다.
5. 그 다음 DPM-Solver++ 공식 update에 더 가까운 logSNR-domain update를 구현한다.

### 5.4 4단계: conservative/aggressive mode 정의

`suggestion.md`의 gate가 선택할 mode를 다음처럼 정의할 수 있다.

```text
conservative: 현재 Euler/Heun, 기존 num_steps 유지
aggressive: DPM-Solver++ 2M 또는 EDM-domain 2차 multistep, 더 적은 NFE
```

예시:

- conservative: Euler 5 steps 또는 Heun 3 steps
- aggressive: DPM-Solver++ 2M 3 steps 또는 2 steps

주의할 점은 현재 DIAMOND 기본값이 이미 `num_steps_denoising=3`으로 작다는 것이다. DPM-Solver가 명확한 speedup을 보이려면 baseline step 수를 늘려 quality를 확보한 뒤 aggressive mode에서 줄이는 실험이 더 설득력 있다.

### 5.5 5단계: probe calibration과 연결

DPM-Solver 자체는 “이 batch가 위험한가”를 판단하는 proxy를 제공하지 않는다. 따라서 PRG에서는 DPM-Solver를 mode로 사용하고, gate input은 TeaCache proxy 또는 별도 drift proxy를 사용한다.

probe label 후보:

- conservative/aggressive rollout의 generated frame L1 또는 perceptual drift
- reward logits drift
- end logits drift
- actor-critic lambda-return drift
- value bootstrap drift

gate output:

```text
if risk_proxy < threshold:
    use aggressive DPM-Solver
else:
    use conservative Euler/Heun
```

## 6. 구현 체크리스트

1. `DiffusionSamplerConfig`에 sampler type과 DPM-Solver 설정을 추가한다.
2. 기존 `sample()`을 `sample_euler_heun()`으로 분리한다.
3. 새 `sample_dpm_solver()` 또는 `DpmSolverSampler`를 추가한다.
4. `Denoiser` wrapper를 만든다.
   - 입력: solver state `x`, scalar sigma 또는 mapped t
   - 출력: data prediction `x0`
5. quantized `denoise()`와 continuous `x0` 경로를 모두 실험 가능하게 한다.
6. `return_denoising_trajectory`가 기존과 같은 shape를 유지하게 한다.
7. `WorldModelEnv` timing metric에 `diffusion_calls`뿐 아니라 model NFE를 추가한다.
8. `quick_run.sh`에서 sampler type별로 metrics summary를 비교할 수 있게 한다.

## 7. 예상 리스크

- 공식 DPM-Solver의 VP schedule과 DIAMOND의 EDM/Karras sigma schedule이 맞지 않아 sample quality가 나빠질 수 있다.
- DIAMOND의 `wrap_model_output()` quantization이 high-order solver의 smoothness assumption을 깨뜨릴 수 있다.
- 현재 3-step baseline에서는 DPM-Solver의 장점이 작거나 불안정할 수 있다.
- actor-critic training에서는 open-loop image metric이 좋아도 return이 떨어질 수 있다.
- aggressive sampler가 reward/end model의 입력 분포를 바꿔 sparse reward game에서 bias를 키울 수 있다.

## 8. 결론

DPM-Solver는 DIAMOND에 “공식 repo 파일 하나를 import해서 끝나는” 형태로 들어가기는 어렵다. 그러나 sampler backend를 분리하고, DIAMOND denoiser를 data prediction model로 감싼 뒤, 2차 multistep solver부터 conservative하게 이식하면 `suggestion.md`의 static faster-sampler baseline과 aggressive mode를 만들 수 있다.

가장 안전한 구현 순서는 다음이다.

1. 기존 Euler/Heun baseline 유지
2. DPM-Solver 설정과 sampler interface 추가
3. DIAMOND sigma-domain 2차 multistep 구현
4. VP adapter 방식은 별도 branch로 검증
5. TeaCache proxy 기반 PRG가 conservative/aggressive sampler를 선택하도록 연결

이 방식이면 DPM-Solver를 DIAMOND의 closed-loop RL training 문제에 맞게 통제 가능한 baseline으로 사용할 수 있다.
