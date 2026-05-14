- Motivation(Problem)
Video world model 을 활용한 RL agent training 은 강력하지만, training loop 안에 video generation inferenece 가 포함된다는 구조적 부담이 있다. 
특히 diffusion 기반 world model 의 inference 비용이 전체 training 시간의 대부분을 차지함. 
Open source model DIAMOND(NIPS’24) 의 RL agent 를 RTX 2080Ti 에서 training 하는 시간 중 75% 이상이 diffusion inference 에 사용됨
RL agent training 1 epoch latency: [actor_critic] total 180.25s | diffusion 136.26s (75.6%, 90.84 ms/step over 1500 calls) | rew_end 16.55s (9.2%, 11.03 ms/step over 1500 calls) | other 27.45s (15.2%)
이게 왜 문제인가?
continual learning, domain randomization 등 adaptive training 이 필요한 상황에서 agent 의 실용성을 저하시킬 수 있음. 
agent model 들이 더욱 high-resolution, long-horizontal video 생성을 요구한다면 world model inference 의 비중이 더욱 커질 것임.

- Related Works:
World model 의 latent embedding 만 사용하여 effective 한 works (v-JEPA, DreamerV3)
하지만 pixel-space world 사용이 더 낫다는 얘기가 있음. [EDELINE(NIPS’25)]

- Our suggestion to address those problems:
SOTA methods for accelerating diffusion inference(DeepCache(CVPR’24) 등?) 를 RL-agent training 상황에 적용하여, RL training accuracy (또는 generated video quality) 를 유지하면서도 더 빠른 training 을 할 수 있도록 함 

- Evaluation Plan
DIAMOND 에서 diffusion 가속 기법 적용 후 FVD 또는 trained agent accuracy 확인
agent training(diffusion inference) latency speedup 확인



참고
- Open source action-video generation model
   - DIAMOND(NIPS’24)
   	- Autoregressive diffusion(world model), reward model, RL agent model training code 포함됨
	- malus02.kaist.ac.kr 서버의 dekim-diamond docker container 에서 돌려봤는데, 직접 들어가서 쓰셔도 될 것 같습니다(수업 서버는 계속 끊기더라고요..)
- Tinyworlds
	- 돌려봤는데 video resolution 이 너무 낮아서 evaluation 하기 힘들 것 같은 느낌
- OASIS
	- 이건 퀄리티가 좋긴 한데 RL agent 용이 아니라 그냥 game engine 이라 DIAMOND 가 더 좋을 것 같았습니다! 

