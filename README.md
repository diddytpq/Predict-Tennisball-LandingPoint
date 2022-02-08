# Predict-Tennisball-LandingPoint
Predict Tennisball  LandingPoint in gazebo

# gazebo 시뮬레이션에서 공 추적 및 낙하지점 예측

## Development environment:

  1. OS : Ubunut 20.04
  2. CUDA : 11.4
  3. GPU : GTX 1880ti
  4. Ros Version : Noetic

You must install ros Noetic before next step

## 1. Setup:

```bash
conda create -n tennis_project python=3.7
conda env update -f environment.yaml

catkin_make
source ./devel/setup.bash
```

## 2. Change python environment path:

```bash
gedit src/ball_description/src/ball_airdynamic.py
```
* Change frist line code  

```bash
#!/home/drcl_yang/anaconda3/envs/tennis_project/bin/python3  >>  #!/home/user_name/your_python_environment_path/ 
```

## 3. Run Demo:

* Start tennis court env in gazebo

```bash
roslaunch mecanum_robot_gazebo tennis_world.launch
```

<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/67572161/150340231-d0544252-7ce5-41a9-9792-74f94f3a1bcc.png"/>
  
</p>

* Run robot tennis match 

```bash
python src/mecanum_robot_gazebo/src/tennis_match.py --mod=0
```
<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/67572161/150342985-c95a5374-c96e-47d2-811c-0695e3465e9b.gif"/>
</p>


## 4. Camera-based ball tracking and predict ball landing point:

```bash
python src/predict_ball_pos/src/predict_ball_landing_point.py
```
<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/67572161/152483052-1c883b3b-7c0b-41f0-89ac-419f4b21a409.gif"/>
</p>


```bash
python src/mecanum_robot_gazebo/src/tennis_match.py --mod=1
```


