# TeaCache 분석 및 DIAMOND 통합 계획

## 1. suggestion.md 기준 목표 이해

`docs/suggestion.md`의 최종 목표는 DIAMOND의 diffusion world model을 단순히 빠르게 만드는 것이 아니라, actor-critic 학습 중 생성되는 imagined rollout의 return utility를 보존하면서 diffusion inference 비용을 줄이는 것이다. 핵심 문제는 open-loop video quality가 유지되어도 action-conditioned transition bias가 closed-loop actor-critic target에 누적될 수 있다는 점이다.

따라서 TeaCache는 이 프로젝트에서 두 가지 역할을 가질 수 있다.

- 첫째, aggressive acceleration 후보이다. denoising step 일부에서 expensive world model forward를 생략하고 cached residual을 재사용해 `diffusion_ms_per_step`과 `actor_critic_total_s`를 낮춘다.
- 둘째, PRG(Probe-Calibrated Rollout Risk Gating)의 cheap risk proxy 후보이다. TeaCache식 relative L1 변화량은 full U-Net output을 계산하기 전에 얻을 수 있으므로, conservative/aggressive mode 선택의 입력 신호로 쓸 수 있다.

중요한 제약은 TeaCache 원 논문과 repo가 주로 DiT/video transformer 구조를 대상으로 한다는 점이다. DIAMOND의 denoiser는 DiT가 아니라 `UNet + AdaGroupNorm` 기반 pixel-space diffusion model이다. 따라서 repo의 코드를 그대로 붙이는 방식이 아니라, TeaCache의 decision rule과 residual reuse pattern을 DIAMOND U-Net 구조에 맞게 이식해야 한다.

## 2. 논문 핵심 정리

TeaCache 논문은 diffusion video generation의 병목을 “모든 denoising timestep에서 전체 모델 출력을 새로 계산한다”는 점으로 본다. 기존 cache 방법이 균일한 timestep 간격으로 output을 재사용하는 것과 달리, TeaCache는 timestep마다 model output 변화량이 균일하지 않다고 보고, output을 계산하기 전 input 쪽 cheap signal로 “이번 step을 계산해야 하는지”를 판단한다.

논문에서 중요한 관찰은 다음과 같다.

- model output 차이는 계산 전에는 알 수 없지만, model input 차이는 이미 존재한다.
- 단순 noisy input이나 timestep embedding만 쓰는 것보다, timestep embedding으로 modulate된 noisy input의 변화량이 output 변화량과 더 잘 맞는다.
- input difference와 output difference 사이에는 scale bias가 있으므로, 모델별 polynomial rescaling으로 보정한다.
- 보정된 relative L1 difference를 누적하다가 threshold를 넘으면 full model forward를 수행하고 cache를 갱신한다. threshold보다 작으면 이전 residual/output을 재사용한다.
- 논문은 Open-Sora-Plan에서 최대 4.41배 speedup과 작은 VBench score 변화라고 보고하지만, 평가는 open-loop video quality 중심이다. DIAMOND에서는 final return과 rollout target drift를 별도로 검증해야 한다.

참고 자료:

- TeaCache 논문: https://arxiv.org/abs/2411.19108
- CVPR 2025 open access PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.pdf
- TeaCache repo: https://github.com/ali-vilab/TeaCache
- DIAMOND 논문: https://arxiv.org/abs/2405.12399

## 3. 로컬 TeaCache 프로젝트 코드 분석

로컬 경로는 `/workspace/TeaCache`이다. repo는 하나의 공통 library라기보다, 여러 diffusion backbone별 forward monkey patch 예제를 모아 둔 구조다.

중요한 파일:

- `TeaCache/README.md`
- `TeaCache/TeaCache4CogVideoX1.5/teacache_sample_video.py`
- `TeaCache/TeaCache4FLUX/teacache_flux.py`
- `TeaCache/TeaCache4Wan2.1/teacache_generate.py`
- `TeaCache/eval/teacache/experiments/*.py`
- `TeaCache/eval/teacache/common_metrics/eval.py`
- `TeaCache/eval/teacache/vbench/cal_vbench.py`

### 3.1 핵심 메커니즘 1: 모델별 polynomial coefficient

`TeaCache/TeaCache4CogVideoX1.5/teacache_sample_video.py`의 `coefficients_dict`가 대표적이다. 모델마다 4차 polynomial coefficient를 두고, relative L1 input change를 output change proxy로 rescale한다.

```python
coefficients_dict = {
    "CogVideoX-2b": [...],
    "CogVideoX-5b": [...],
    "CogVideoX1.5-5B": [...],
}
```

이 coefficient는 DIAMOND에 그대로 재사용하면 안 된다. DIAMOND는 architecture, input scale, timestep parameterization, action conditioning이 모두 다르므로, DIAMOND 전용 calibration이 필요하다.

### 3.2 핵심 메커니즘 2: cheap proxy 계산

CogVideoX 예제에서는 timestep embedding을 만든 뒤 `emb`의 relative L1 변화량을 계산한다.

```python
rescale_func = np.poly1d(self.coefficients)
self.accumulated_rel_l1_distance += rescale_func(
    ((emb - self.previous_modulated_input).abs().mean()
     / self.previous_modulated_input.abs().mean()).cpu().item()
)
```

FLUX/Mochi/Lumina 계열 예제에서는 첫 transformer block의 norm/modulation 결과인 `modulated_inp`를 사용한다. 이쪽이 논문 아이디어에 더 가깝다.

```python
modulated_inp, ... = self.transformer_blocks[0].norm1(inp, emb=temb_)
rel_l1 = (modulated_inp - previous_modulated_input).abs().mean() / previous_modulated_input.abs().mean()
```

DIAMOND에서는 이에 대응되는 cheap proxy 후보가 세 가지다.

- `cond = cond_proj(noise_emb(c_noise) + act_emb(act))`
- `x_in = conv_in(cat(obs, noisy_next_obs))`
- 첫 `ResBlock.norm1(x_in, cond)`의 output

세 번째가 TeaCache 논문 취지와 가장 가깝지만, 구현 난도가 조금 높다. 첫 버전은 `cond`와 `x_in`의 normalized L1을 함께 기록하고, calibration 결과에 따라 proxy를 선택하는 것이 안전하다.

### 3.3 핵심 메커니즘 3: 누적 threshold와 refresh

TeaCache는 매 timestep에서 즉시 threshold를 판단하지 않고, 보정된 relative L1을 누적한다.

```python
if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
    should_calc = False
else:
    should_calc = True
    self.accumulated_rel_l1_distance = 0
```

또한 첫 step과 마지막 step은 full 계산을 강제한다.

```python
if self.cnt == 0 or self.cnt == self.num_steps - 1:
    should_calc = True
    self.accumulated_rel_l1_distance = 0
```

DIAMOND의 기본 `world_model_env.diffusion_sampler.num_steps_denoising`은 3이다. 이 설정에서는 첫/마지막 full 계산을 강제하면 skip 가능한 step이 거의 없어 speedup 상한이 낮다. TeaCache 실험을 하려면 다음 중 하나가 필요하다.

- baseline denoising step을 5, 8, 10 등으로 올려 quality/latency tradeoff를 만든다.
- aggressive mode에서는 마지막 step full 계산 강제를 완화하거나, threshold 정책을 DIAMOND용으로 새로 둔다.
- TeaCache를 단독 speedup 수단보다 PRG의 risk proxy로 우선 사용한다.

### 3.4 핵심 메커니즘 4: residual reuse

CogVideoX 예제는 transformer blocks 전체의 residual을 저장한다.

```python
ori_hidden_states = hidden_states.clone()
...
hidden_states, encoder_hidden_states = block(...)
...
self.previous_residual = hidden_states - ori_hidden_states
```

skip할 때는 full transformer blocks를 실행하지 않고 residual만 더한다.

```python
hidden_states += self.previous_residual
encoder_hidden_states += self.previous_residual_encoder
```

DIAMOND U-Net에서도 가장 자연스러운 이식 방식은 `UNet` 전체를 expensive block으로 보고 다음 residual을 저장하는 것이다.

```python
x_in = conv_in(cat(obs, noisy_next_obs))
x_unet = unet(x_in, cond)
previous_unet_residual = x_unet - x_in
```

skip 시:

```python
x_unet = x_in + previous_unet_residual
```

그 뒤 `norm_out`과 `conv_out`은 항상 실행한다. 이 부분은 상대적으로 작고, output shape 안정성을 유지하는 데 유리하다.

### 3.5 평가 코드

TeaCache repo의 `eval/teacache`는 VBench, LPIPS, PSNR, SSIM 평가를 포함한다. `common_metrics/eval.py`는 original model video를 gt로 두고 generated video와 비교한다. DIAMOND에는 바로 맞지 않는다. DIAMOND의 main metric은 다음 순서로 둬야 한다.

- `diffusion_ms_per_step`: 낮을수록 좋음
- `actor_critic_total_s`: 낮을수록 좋음
- `final_return_mean`: baseline 대비 유지되어야 함
- FVD 또는 rollout video metric: 보조 지표
- probe target drift: PRG calibration용 핵심 지표

## 4. DIAMOND 쪽 관련 코드

DIAMOND에서 TeaCache를 넣을 후보 경로는 다음이다.

- `src/models/diffusion/diffusion_sampler.py`
  - `DiffusionSampler.sample()`이 denoising loop를 돈다.
  - 현재 각 sigma마다 `self.denoiser.denoise(...)`를 호출한다.
- `src/models/diffusion/denoiser.py`
  - `Denoiser.denoise()`가 `compute_conditioners`, `compute_model_output`, `wrap_model_output`을 호출한다.
  - `compute_model_output()`에서 `InnerModel`이 실제 expensive forward를 수행한다.
- `src/models/diffusion/inner_model.py`
  - `InnerModel.forward()`에서 `cond`, `conv_in`, `unet`, `norm_out`, `conv_out` 순서로 실행된다.
- `src/models/blocks.py`
  - `UNet`, `ResBlock`, `AdaGroupNorm`이 있다. TeaCache proxy를 더 정교하게 만들려면 이 파일의 `UNet.forward()` 또는 첫 `ResBlock` 접근이 필요하다.
- `src/envs/world_model_env.py`
  - actor-critic rollout에서 diffusion inference timing이 이미 계측된다.

## 5. DIAMOND 통합 계획

### 5.1 1단계: 계측 우선 추가

가장 먼저 TeaCache를 적용하지 않고 proxy만 기록한다.

구현 위치:

- `src/models/diffusion/inner_model.py`
- 또는 새 파일 `src/models/diffusion/teacache.py`

기록할 값:

- `cond_rel_l1`
- `conv_in_rel_l1`
- 가능하면 `first_adagn_rel_l1`
- 실제 `model_output_rel_l1`
- 실제 `unet_residual_rel_l1`
- timestep index, sigma, action change, batch id

목적:

- DIAMOND에서도 TeaCache proxy와 output drift 사이 상관이 있는지 확인한다.
- polynomial coefficient를 DIAMOND용으로 fit한다.
- proxy가 game/action dynamics에 따라 얼마나 흔들리는지 본다.

### 5.2 2단계: U-Net residual cache 구현

새 config 후보:

```yaml
world_model_env:
  diffusion_sampler:
    teacache:
      enabled: false
      rel_l1_thresh: 0.1
      force_first: true
      force_last: true
      proxy: conv_in
      coefficients: null
      reset_each_sample: true
```

구현 방식:

- `DiffusionSampler.sample()` 시작 시 cache state를 reset한다.
- `Denoiser.denoise()`에 optional cache context를 넘긴다.
- `InnerModel.forward()` 또는 별도 `forward_teacache()`에서 `x_in`, `cond`를 만든 뒤 `should_calc`를 판단한다.
- full 계산이면 `self.unet(x_in, cond)` 실행 후 `previous_unet_residual = x_unet - x_in` 저장.
- skip이면 `x_unet = x_in + previous_unet_residual`.
- 이후 `norm_out`, `conv_out`, `wrap_model_output`은 기존 경로를 유지한다.

주의:

- cache state는 rollout batch 간 공유하면 안 된다. diffusion trajectory 내부에서만 유지해야 한다.
- actor-critic training은 batch마다 `prev_obs`, `prev_act`가 다르므로 global module attribute에 cache를 두면 DDP/compile과 충돌할 수 있다. sampler-local cache object를 넘기는 방식이 더 안전하다.
- `torch.compile` 사용 시 동적 branch가 compile graph를 깨거나 느리게 만들 수 있다. TeaCache 실험에서는 먼저 `training.compile_wm=false`로 baseline을 잡는 것이 좋다.

### 5.3 3단계: PRG와 연결

`suggestion.md`의 핵심은 static TeaCache가 아니라 risk-gated acceleration이다. TeaCache proxy는 다음처럼 gate input으로 쓴다.

```text
risk_proxy = a * accumulated_rel_l1 + b * sigma_gap + c * action_change + d
```

초기에는 learned model 없이 threshold rule로 시작한다.

- low risk: TeaCache aggressive threshold 사용
- high risk: TeaCache off 또는 conservative threshold 사용

그 다음 probe calibration을 추가한다.

- 전체 imagined batch 중 일부만 conservative/aggressive 둘 다 rollout한다.
- 두 rollout의 lambda-return, reward/end prediction, value bootstrap 차이를 label로 만든다.
- TeaCache proxy가 이 label을 잘 예측하도록 threshold를 calibration game에서 고정한다.
- reporting game에서는 threshold를 바꾸지 않는다.

### 5.4 4단계: 실험 우선순위

첫 구현 후 바로 볼 지표:

- `actor_critic_total_s`
- `diffusion_total_s`
- `diffusion_ms_per_step`
- TeaCache skip ratio
- `final_return_mean`

그 다음 볼 지표:

- conservative vs TeaCache rollout frame drift
- reward/end drift
- lambda-return drift
- FVD 또는 간단한 frame metric

## 6. 예상 리스크

- DIAMOND 기본 denoising step이 너무 작아 TeaCache 단독 speedup이 작을 수 있다.
- DiT용 coefficient를 재사용하면 proxy가 잘못 calibration될 가능성이 높다.
- U-Net residual reuse가 final return에는 open-loop frame metric보다 더 민감할 수 있다.
- cache branch가 `torch.compile`과 충돌하면 speedup이 상쇄될 수 있다.
- aggressive skip은 sparse reward game에서 작은 visual/transition bias를 크게 증폭시킬 수 있다.

## 7. 결론

TeaCache는 DIAMOND에 “drop-in 코드”로 붙이기보다, `InnerModel`의 U-Net residual reuse와 PRG risk proxy로 재해석해 이식하는 것이 맞다. 최우선 구현은 full TeaCache가 아니라 proxy logging과 DIAMOND 전용 coefficient calibration이다. 그 다음 sampler-local cache state를 넣고, 마지막으로 `suggestion.md`의 probe-calibrated gating과 연결하는 순서가 안전하다.
