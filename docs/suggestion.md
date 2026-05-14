# Probe-Calibrated Rollout Risk Gating for Return-Preserving Diffusion Acceleration


## Introduction (3 bullets)
- video world model 기반 RL의 broader problem은 high-fidelity imagined environment가 agent return을 끌어올릴 수 있음에도, training loop 내부에서 반복되는 diffusion inference 때문에 실제 wall-clock 학습이 과도하게 느려진다는 점이다.
- 기존 diffusion acceleration 연구는 `FID`, `VBench`, human preference 같은 open-loop perceptual quality가 유지되면 downstream utility도 유지된다고 암묵적으로 가정하고, 기존 world model RL 연구는 반대로 simulator fidelity가 고정돼 있다고 놓고 policy update만 분석하는 경향이 있다.
- 그러나 RL training 실제 운영에서는 action-conditioned rollout이 actor-critic target을 계속 다시 만들기 때문에, perceptual quality가 유지돼도 작은 transition bias가 closed-loop로 누적되며, 특히 late training의 sharp policy 구간에서는 final return과 sample efficiency가 비선형적으로 하락할 수 있다.


## Background (3 bullets)
- `DIAMOND`는 diffusion world model이 Atari 100k에서 strong RL utility를 낼 수 있음을 보였지만, 1 epoch latency `180.25s` 중 diffusion inference가 `136.26s (75.6%)`를 차지해 faster sampler가 아니라면 systems advantage가 빠르게 상쇄되는 구조를 드러냈다.
- `DPM-Solver++`와 `TeaCache`는 각각 retraining 없는 step reduction과 adaptive cache reuse로 큰 speedup을 달성한 가장 직접적인 baseline이지만, 평가는 주로 open-loop generation quality와 throughput에 머물러 있고 `return-preserving speedup`은 검증하지 않았다.
- `DreamerV3`, `DayDreamer`, `STORM`, `IRIS`는 learned world model의 핵심 objective가 pixel realism이 아니라 wall-clock learning utility, sample efficiency, downstream control robustness임을 보여주므로, 우리 문제를 단순 acceleration이 아니라 `training utility under fixed wall-clock` 문제로 재정의할 근거를 제공한다.


## Problem Definition (3 bullets)
- `DIAMOND`류 workload의 병목은 단순히 diffusion 생성이 느리다는 사실이 아니라, actor-critic update를 위해 action-conditioned imagined rollout을 반복 생성하는 training loop 구조다. 대표 측정치로 1 epoch당 약 `1500`회의 diffusion call과 평균 `90.84 ms/step`이 누적돼 전체 학습 시간의 `75.6%`를 점유한다.
- 기존 정적 acceleration은 이 workload에서 두 방식으로 실패하거나 비효율적이다. `low-step solver`는 모든 state에 걸쳐 작은 transition bias를 전역적으로 주입하는 `global under-resolution` 위험이 있고, `cache reuse`는 대부분의 step에서는 안전하지만 motion 변화나 action consequence가 큰 구간에서만 급격히 오류가 커지는 `state-dependent spike` 위험이 있다. 따라서 perceptual metric이 맞아도 `trained agent return`은 다르게 무너질 수 있다.
- 이 문제는 자명하지 않다. acceleration 논문은 `VBench`류 open-loop metric으로 충분하다고 보고, world model RL 논문은 full-fidelity simulator를 전제로 두어 왔기 때문에, `quality 유지 but return 하락`, `training stage별 민감도 변화`, `method family별 closed-loop risk signature`를 같은 프로토콜 아래에서 측정한 연구가 아직 없다.


## Proposed Idea (2-4 bullets)
- `DIAMOND` training loop에 `Probe-Calibrated Rollout Risk Gating (PRG)`를 삽입한다. 각 imagined rollout batch는 두 mode 중 하나를 선택한다: `conservative = DPM-Solver++ with moderate step count and no cache reuse`, `aggressive = low-step DPM-Solver++ + TeaCache`. 핵심은 더 빠른 새 accelerator를 만드는 것이 아니라, closed-loop risk가 낮은 batch에만 aggressive mode를 허용하는 `training-aware fidelity allocation`이다.
- gate는 noisy RL statistic 대신 `DIAMOND`에서 직접 읽을 수 있는 cheap signal만 사용한다. `rollout depth h/H`, `normalized policy entropy`, 그리고 TeaCache식 `timestep/action-conditioned proxy margin`을 계산해 1차 risk proxy를 만든다. 여기서 proxy margin은 expensive U-Net pass 전에 이미 존재하는 timestep embedding과 noisy latent/action conditioning 차이로 추정하며, 값이 작을수록 cached high-level feature reuse가 안전하다고 본다.
- 전체 imagined batch의 `1-5%`에서만 `probe`를 수행해 같은 initial state/action minibatch를 conservative/aggressive 두 mode로 모두 굴리고, `n-step bootstrap target drift`, `reward disagreement`, `action agreement`로부터 `unsafe` label을 만든다. held-out calibration game subset에서 risk proxy를 이 label에 맞춰 한 번 calibrate하고, reporting game에서는 threshold를 고정한다. 이렇게 하면 controller 입력은 heuristic RL instability signal이 아니라 `direct control drift`에 맞춰 보정된다는 점이 기존 adaptive schedule과 근본적으로 다르다.
- 핵심 training-time hypothesis는 두 가지다. `1)` open-loop visual quality가 matched되어도 closed-loop target drift는 다를 수 있다. `2)` aggressive acceleration을 early training과 low-risk rollout에 집중하고 late training/high-risk rollout에서는 보수적으로 되돌리면, matched-speed static hybrid나 hand-crafted stage schedule보다 더 나은 `return-speed Pareto frontier`를 만들 수 있다. 구현은 sampler wrapper와 cache instrumentation만 수정하면 되어 world model 재학습이 필요 없다.


## Evaluation Plan (4-6 bullets)
- 비교 baseline은 `vanilla DIAMOND`, 정적 `DPM-Solver++`, 정적 `TeaCache`, 제안법과 동일한 평균 speedup을 맞춘 `static hybrid`, `early-aggressive/late-conservative` hand-crafted stage schedule, 그리고 hindsight probe로 mode를 고르는 `oracle-lite` upper bound로 둔다. `IRIS`와 `STORM`은 direct acceleration baseline이 아니라 efficiency reference로 분리 보고한다.
- 핵심 metric은 `net epoch latency`, `imagined rollout throughput`, `final human normalized score`, `learning-curve AUC`, `equal wall-clock budget return`, `equal interaction budget return`, `seed variance`다. 보조로 `FVD` 또는 perceptual score를 보고하되, main success criterion은 `>=1.4x` net speedup을 달성하면서 `equal interaction budget` 기준 final return drop을 `<=5%`로 제한하거나, 동일 wall-clock에서 더 높은 return을 내는지로 고정한다.
- 중요한 ablation은 `depth-only / entropy-only / proxy-only / probe-calibrated full signal`, `probe budget (0%, 1%, 5%)`, `solver step count`, `TeaCache threshold`, `rollout horizon`, `training stage`, `calibration split size`, `matched average speedup level`이다. 이를 통해 controller가 정말 필요한지, 아니면 단순 stage schedule만으로 충분한지를 분리한다.
- failure case 검증은 두 축으로 설계한다. 첫째, matched perceptual score를 갖는 `solver-only`와 `TeaCache` 설정을 골라 `return`, `action agreement`, `n-step target drift`가 다르게 나오는지 측정해 metric mismatch를 드러낸다. 둘째, 같은 평균 speedup을 갖는 `static hybrid`와 `PRG`를 비교해, 제안법의 이득이 단순히 덜 가속해서 생긴 보수성인지 아니면 better allocation 때문인지 확인한다.
- 실험 환경은 single-GPU `RTX 3090` 또는 `RTX 2080 Ti`에서 Atari 100k의 `5-10`개 게임을 사용하고, sparse reward, flicker sensitivity, fast dynamics가 섞이도록 선택한다. calibration용 game과 최종 reporting game을 분리하고, 각 설정은 `3-5` seeds로 반복한다.
- 시스템 보고 항목으로 `controller overhead`, `peak memory`, `cache hit ratio`, `mode selection frequency`, `probe cost`를 반드시 포함한다. probe는 전체 imagined batch의 `5%` 이하, bookkeeping overhead는 epoch time의 `2%` 이하를 목표로 하며, 모든 speedup 수치는 이 overhead를 포함한 `net utility` 기준으로만 보고한다.


## 결정 로그
| 검토한 대안 | 채택/기각 | 이유 | 관련 리뷰 피드백 |
|------------|----------|------|----------------|
| `../docs/research_background/INDEX.md` 기반 background 정리 | 기각 | 해당 경로가 비어 있어 지시된 `INDEX.md`를 읽을 수 없었고, 대신 `../docs/papers/INDEX.md`와 20편 paper summary를 primary background로 사용했다 | 두 리뷰 모두 동일 사실을 명시했고, v3에서도 source limitation을 투명하게 기록 |
| `TeaCache`를 DIAMOND에 고정 적용하는 단순 drop-in | 기각 | 속도는 줄일 수 있어도 `quality 유지 but return 하락`과 training-stage sensitivity를 설명하지 못해 연구 가설이 약하다 | v1, v2 공통 반복 지적: 단순 acceleration 적용을 넘어서는 training-time insight가 필요 |
| `TeaCache`와 `FasterCache`를 동시에 primary cache baseline으로 두는 hybrid | 기각 | `FasterCache`는 CFG redundancy 가정이 강하고, `DIAMOND` 이식 가능성 설명이 약해 구현/해석 복잡도만 커진다 | 최신 리뷰 최종 권고 3, 개선 제안 1: `TeaCache/FasterCache` 이식 가능성을 더 구체화하고 하나만 primary로 고정하라는 지적 반영 |
| `TD-error`, `advantage variance` 중심의 linear risk score | 기각 | acceleration risk의 원인이라기보다 training instability의 결과일 가능성이 커서 confounded signal이 되기 쉽다 | v1 약점 2, v2 최종 권고 1: `risk score`의 입력 신호와 causal 정당화를 더 명확히 하라는 반복 지적 반영 |
| `probe-calibrated direct control drift`로 gate를 보정하는 PRG | 채택 | heuristic schedule이 아니라 closed-loop target drift에 정렬된 reusable decision rule을 만들 수 있어, controller 필요성과 signal 타당성을 동시에 검증할 수 있다 | 최신 리뷰 최종 권고 1, 2를 직접 반영 |
| `hand-crafted stage schedule`을 main method로 사용 | 기각 | early/aggressive, late/conservative 직관은 강하지만 game마다 최적 전환점이 달라 adaptive controller의 필요성을 증명하지 못한다 | 최신 리뷰 최종 권고 2: `controller 필요성`을 증명하려면 이 baseline을 추가하되 main method로 두지 말라는 지적 반영 |
| `few-step student`나 distillation 계열을 v3 main scope에 포함 | 기각 | 추가 teacher/student preparation cost와 calibration 난도 때문에 clean comparison이 깨지고, current research question은 `training-free net utility`만으로도 충분히 강하다 | v1 최종 권고 2의 scope 축소 지적과 v2 반복 약점의 연장선 반영 |

