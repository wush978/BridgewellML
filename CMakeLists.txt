cmake_minimum_required (VERSION 2.8.11)
project (BridgewellML)

add_subdirectory(test)

enable_testing()
add_test( NAME logisticregression COMMAND logisticregression.exe )
