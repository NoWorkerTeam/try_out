"""
torch - 강력한 GPU 지원 기능을 갖춘 Numpy와 같은 라이브러리
torch.autograd - Torch에서 모든 차별화된 Tensor 작업을 지원하는 테이프 기반 자동 미분화 라이브러리
torch.optim	- SGD, RMSProp, LBFGS, Adam 등과 같은 표준 최적화 방법으로 torch.nn과 함께 사용되는 최적화 패키지
torch.nn - 최고의 유연성을 위해 설계된 자동 그래프와 깊이 통합된 신경 네트워크 라이브러리
torch.legacy(.nn/optim)	- 이전 버전과의 호환성을 위해 Torch에서 이식된 레거시 코드
torch.utils	- 편의를 위해 DataLoader, Trainer 및 기타 유틸리티 기능
torch.multiprocessing - 파이썬 멀티 프로세싱을 지원하지만, 프로세스 전반에 걸쳐 Torch Tensors의 마법같은 메모리 공유 기능을 제공.
데이터 로딩 및 호그 워트 훈련에 유용

cuda - gpu 병렬처리 라이브러리
"""


### main.py args 설명

if __name__ == '__main__':
    argparse = 1 # 임의로 설정(신경x)
    args = argparse.ArgumentParser()

### 아래의 수치값 조정을 통해 모델 성능 향상 ###

    # DONOTCHANGE
    args.add_argument('--mode', type=str, default='train') # train: 훈련 모드 / test: 테스트 모드
    args.add_argument('--iteration', type=str, default='0') # 훈련 시작 시점의 반복 횟수
    args.add_argument('--pause', type=int, default=0) # 훈련 중지 후 다시 시작할 때 사용할 반복 횟수

    # Parameters 
    args.add_argument('--use_cuda', type=bool, default=True) # GPU 사용 여부
    args.add_argument('--seed', type=int, default=777) # 난수 생성 시드
    args.add_argument('--num_epochs', type=int, default=10) # 훈련 에포크 수
    args.add_argument('--batch_size', type=int, default=128) # 배치 크기
    args.add_argument('--save_result_every', type=int, default=10) # 몇 에포크마다 결과를 저장할지
    args.add_argument('--checkpoint_every', type=int, default=1) # 몇 에포크마다 체크포인트를 저장할지
    args.add_argument('--print_every', type=int, default=50) # 몇 배치마다 로그를 출력할지
    args.add_argument('--dataset', type=str, default='kspon') # 사용할 데이터셋
    args.add_argument('--output_unit', type=str, default='character') # 출력 단위 (문자 또는 음소)
    args.add_argument('--num_workers', type=int, default=8) # 데이터 로더 스레드 수
    args.add_argument('--num_threads', type=int, default=16) # PyTorch 스레드 수
    args.add_argument('--init_lr', type=float, default=1e-06) # 초기 학습률
    args.add_argument('--final_lr', type=float, default=1e-06) # 최종 학습률
    args.add_argument('--peak_lr', type=float, default=1e-04) # 최대 학습률
    args.add_argument('--init_lr_scale', type=float, default=1e-02) # 초기 학습률 스케일
    args.add_argument('--final_lr_scale', type=float, default=5e-02) # 최종 학습률 스케일
    args.add_argument('--max_grad_norm', type=int, default=400) # 그래디언트 클리핑 최대값
    args.add_argument('--warmup_steps', type=int, default=1000) # 워밍업 스텝 수
    args.add_argument('--weight_decay', type=float, default=1e-05) # 가중치 감소 계수
    args.add_argument('--reduction', type=str, default='mean') # 손실 함수의 손실값 축소 방식 (mean 또는 sum)
    args.add_argument('--optimizer', type=str, default='adam') # 사용할 최적화기 (adam 또는 sgd)
    args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler') # 사용할 학습률 스케줄러 (tri_stage_lr_scheduler 또는 onecyclelr_scheduler)
    args.add_argument('--total_steps', type=int, default=200000) # 총 훈련 스텝 수

    args.add_argument('--architecture', type=str, default='deepspeech2') # 사용할 모델 구조 (deepspeech2 또는 transformer)
    args.add_argument('--use_bidirectional', type=bool, default=True) # 양방향 RNN 사용 여부
    args.add_argument('--dropout', type=float, default=3e-01) # 드롭아웃 비율
    args.add_argument('--num_encoder_layers', type=int, default=3) # 인코더 레이어 수
    args.add_argument('--hidden_dim', type=int, default=1024) # 은닉층 크기
    args.add_argument('--rnn_type', type=str, default='gru') # RNN 유형 (gru 또는 lstm)
    args.add_argument('--max_len', type=int, default=400) # 입력 시퀀스의 최대 길이
    args.add_argument('--activation', type=str, default='hardtanh') # 활성화 함수 (hardtanh 또는 relu)
    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0) # Teacher forcing 비율
    args.add_argument('--teacher_forcing_step', type=float, default=0.0) # Teacher forcing 감소 스케일
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0) # Teacher forcing 최소 비율
    args.add_argument('--joint_ctc_attention', type=bool, default=False) # CTC와 어텐션을 함께 사용할지 여부

    args.add_argument('--audio_extension', type=str, default='pcm') # 오디오 파일 확장자
    args.add_argument('--transform_method', type=str, default='fbank') # 오디오 전처리 방법 (fbank 또는 spectogram)
    args.add_argument('--feature_extract_by', type=str, default='kaldi') # 특징 추출 라이브러리 (kaldi 또는 torchaudio)
    args.add_argument('--sample_rate', type=int, default=16000) # 샘플링 레이트
    args.add_argument('--frame_length', type=int, default=20) # 프레임 길이
    args.add_argument('--frame_shift', type=int, default=10) # 프레임 이동 간격
    args.add_argument('--n_mels', type=int, default=80) # Mel filter banks 수
    args.add_argument('--freq_mask_para', type=int, default=18) # 주파수 마스킹 매개변수
    args.add_argument('--time_mask_num', type=int, default=4) # 시간 마스킹 개수
    args.add_argument('--freq_mask_num', type=int, default=2) # 주파수 마스킹 개수
    args.add_argument('--normalize', type=bool, default=True) # 특징값 정규화 여부
    args.add_argument('--del_silence', type=bool, default=True) # 정적 소음 제거 여부
    args.add_argument('--spec_augment', type=bool, default=True) # SpecAugment 적용 여부
    args.add_argument('--input_reverse', type=bool, default=False) # 입력 시퀀스 반전 여부

## 명령어 설명
# ls : 현재 폴더 내 파일 내용 확인 ex) ls
# cd : '특정'폴더로 이동 ex) cd baseline / cd .. : 상위 폴더로 이동
# cat : 파일 전체 내용 출력 ex) cat main.py

## 명령어 도움말
# --help ex) nova run --help

## 모델 현황 파악
# nova ps -a : 세션에 았는 모든 모델 출력
# nova ps --only-running : 학습 진행중인 모델만 출력

## 생성된 세션 삭제
# nova rm ex) nova rm junehong2/Tr1Ko/40

## arg 값 조정방식
# nova run -d Tr1Ko -a "--dropout=0.4" -a "--num_encoder_layers=5"

## 명칭이 'Tr1Ko'인 데이터셋을 사용해 세션 실행하기
# nova run -d Tr1Ko

## 2GPU와 16CPU, 160GB 메모리를 사용하여 세션 실행하기   
# nova run -d Tr1Ko -g 2 -c 16 --memory 160G  

## 세션 로그 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조 ex) junehong2/Tr1Ko/40
# nova logs [세션명] ex) nova logs junehong2/Tr1Ko/40

## 모델 목록 및 제출하고자 하는 모델의 checkpoint별 summary 
# nova model ls [세션명] ex) nova model ls junehong2/Tr1Ko/40

## 모델 제출 전 제출 코드에 문제가 없는지 점검하기('-t' 옵션)
# nova submit -t [세션명] [모델_checkpoint_번호]

## 모델 제출하기
## 제출 후 리더보드에서 점수 확인 가능
# nova submit [세션명] [모델_checkpoint_번호]

