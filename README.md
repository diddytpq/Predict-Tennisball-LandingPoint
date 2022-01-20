# Predict-Tennisball-LandingPoint
Predict Tennisball  LandingPoint in gazebo

# gazebo 시뮬레이션에서 공 추적 및 낙하지점 예측

## Development environment:

OS : Ubunut 20.04
CUDA :
Graphic Card : GTX 1880ti
Ros Version : noetic

You must install ros before next step

## First time setup:

```bash
conda create -n tennis_project python=3.7
conda env update -n tennis_project -f env.yml

catkin_make
source ./devel/setup.bash
```

### air dynamic.py 

## Run Demo:

start tennis court env in gazebo

```bash
roslaunch mecanum_robot_gazebo tennis_ver2.launch
```

<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/67572161/150340231-d0544252-7ce5-41a9-9792-74f94f3a1bcc.png"/>
  
</p>

this command begin robot tennis match 
```bash
python src/mecanum_robot_gazebo/src/tennis_match_ver3.py
```
<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/67572161/150342985-c95a5374-c96e-47d2-811c-0695e3465e9b.gif"/>
</p>




## Camera-based ball tracking installed on the net post

```bash
python src/predict_ball_pos/src/predict_ball_position.py
```
#### 카메라 화면과 테니스 코트 탑뷰 이미지 추가

```bash
python src/mecanum_robot_gazebo/src/tennis_match_ver3.py --mod=1
```
