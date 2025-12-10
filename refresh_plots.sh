#!/bin/bash
python ./graph_algorithm.py one_car_2
python ./graph_algorithm.py one_car_6
python ./graph_algorithm.py three_cars_1_cyan_6_purple_8_white
python ./graph_algorithm.py two_cars_6_cyan_5_black
python ./graph_algorithm.py two_cars_6_green_8_black

python ./plot_final_trajectory.py one_car_2
python ./plot_final_trajectory.py one_car_6
python ./plot_final_trajectory.py three_cars_1_cyan_6_purple_8_white
python ./plot_final_trajectory.py two_cars_6_cyan_5_black
python ./plot_final_trajectory.py two_cars_6_green_8_black

python ./plot_final_trajectory.py one_car_2 -g
python ./plot_final_trajectory.py one_car_6 -g
python ./plot_final_trajectory.py three_cars_1_cyan_6_purple_8_white -g
python ./plot_final_trajectory.py two_cars_6_cyan_5_black -g
python ./plot_final_trajectory.py two_cars_6_green_8_black -g

cp demos/one_car_2/graph_alg_plot.png docs/assets/img/one_car_2_graph.png
cp demos/one_car_2/trajectory_plot.png docs/assets/img/one_car_2_KF.png

cp demos/one_car_6/graph_alg_plot.png docs/assets/img/one_car_6_graph.png
cp demos/one_car_6/trajectory_plot.png docs/assets/img/one_car_6_KF.png

cp demos/three_cars_1_cyan_6_purple_8_white/graph_alg_plot.png docs/assets/img/three_cars_1_cyan_6_purple_8_white_graph.png
cp demos/three_cars_1_cyan_6_purple_8_white/trajectory_plot.png docs/assets/img/three_cars_1_cyan_6_purple_8_white_KF.png

cp demos/two_cars_6_cyan_5_black/graph_alg_plot.png docs/assets/img/two_cars_6_cyan_5_black_graph.png
cp demos/two_cars_6_cyan_5_black/trajectory_plot.png docs/assets/img/two_cars_6_cyan_5_black_KF.png

cp demos/two_cars_6_green_8_black/graph_alg_plot.png docs/assets/img/two_cars_6_green_8_black_graph.png
cp demos/two_cars_6_green_8_black/trajectory_plot.png docs/assets/img/two_cars_6_green_8_black_KF.png
