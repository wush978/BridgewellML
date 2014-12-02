cmake_minimum_required (VERSION 2.8.11)
project (BridgewellML CXX)



add_subdirectory(test)

enable_testing()
add_test( NAME logisticregression COMMAND logisticregression.exe )
add_test( NAME logisticregressionpoly2 COMMAND logisticregressionpoly2.exe )
add_test( NAME neuronnetwork COMMAND neuronnetwork.exe )
add_test( NAME censoredregression COMMAND censoredregression.exe )
