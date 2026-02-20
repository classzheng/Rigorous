/******************************************************************************
 * Rigorous/Rigorous: Template file module of the library.                    *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.20 (latest upd)                                              *
 * @Description: The includeing file of #Rigorous Lib files.                  *
 * @Modules: {}                                                               *
 ******************************************************************************/
 
#pragma once
#pragma GCC optimize (2)

#if defined($DISABLE_ALL)
#   pragma message("Disabled #Rigorous.")
#else
#	if !defined($DISABLE_ANN)
#		include "neuralnetwork.hpp"
#	else
#   	pragma message("Disabled #Rigorous/NeuralNetwork.")
#   endif

#	if !defined($DISABLE_SVM)
#		include "c-svm.hpp"
#	else
#   	pragma message("Disabled #Rigorous/C-SVM.")
#   endif

#	if !defined($DISABLE_RF)
#		include "randomforest.hpp"
#	else
#   	pragma message("Disabled #Rigorous/RandomForest.")
#   endif

#	if defined($USING_PACKAGES)
	using namespace BackwardGrad;
	using namespace NeuralNetwork;
	using namespace SVMPackage;
	using namespace RandomForest;
#	endif

#endif

