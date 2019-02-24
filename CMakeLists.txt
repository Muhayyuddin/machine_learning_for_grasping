cmake_minimum_required(VERSION 2.8.3)
project(machine_learning)

find_package(catkin REQUIRED 
COMPONENTS 
message_generation 
std_msgs 
geometry_msgs)


add_service_files
(
DIRECTORY srv 
FILES 
grasp_feasibility_evaluation.srv
)

generate_messages
(
DEPENDENCIES
std_msgs 
geometry_msgs
)

catkin_package
(
CATKIN_DEPENDS 
message_runtime 
std_msgs 
geometry_msgs
)


catkin_install_python(PROGRAMS
  scripts/machine_learning_server
  scripts/machine_learning_client
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)


