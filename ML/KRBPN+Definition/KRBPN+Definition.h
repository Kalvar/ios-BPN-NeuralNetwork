//
//  KRBPN+Definition.h
//  BPN
//
//  Created by Kalvar Lin on 2015/8/22.
//  Copyright (c) 2015年 Kuo-Ming Lin. All rights reserved.
//

#import "KRQuickProp.h"
#import "KRBPN+NSUserDefaults.h"

/*
 * @ 當前訓練的 BPN Network 數據資料
 *   - trainedInfo = @{};
 *      - KRBPNTrainedInputWeights      : NSMutableArray, 調整後的輸入層各向量值到隱藏層神經元的權重
 *      - KRBPNTrainedHiddenWeights     : NSMutableArray, 調整後的隱藏層神經元到輸出層神經元的權重值
 *      - KRBPNTrainedHiddenBiases      : NSMutableArray, 調整後的隱藏層神經元的偏權值
 *      - KRBPNTrainedOutputBiases      : NSMutableArray, 調整後的輸出層神經元偏權值
 *      - KRBPNTrainedOutputResults     : NSArray,        輸出結果
 *      - KRBPNTrainedIterations        : NSInteger,      已訓練到第幾代
 *
 */
static NSString *KRBPNTrainedInputWeights   = @"KRBPNTrainedInputWeights";
static NSString *KRBPNTrainedHiddenWeights  = @"KRBPNTrainedHiddenWeights";
static NSString *KRBPNTrainedHiddenBiases   = @"KRBPNTrainedHiddenBiases";
static NSString *KRBPNTrainedOutputBiases   = @"KRBPNTrainedOutputBiases";
static NSString *KRBPNTrainedOutputResults  = @"KRBPNTrainedOutputResults";
static NSString *KRBPNTrainedIterations     = @"KRBPNTrainedIterations";

typedef enum KRBPNActivations
{
    //Sigmoid (Normally usage)
    KRBPNActivationBySigmoid = 0,
    //Tanh (Better usage, but it need to properly design training-patterns and initial weights)
    KRBPNActivationByTanh,
    //Fuzzy, still not complete since this method needs to calculate the relationships of each other.
    KRBPNActivationByFuzzy
}KRBPNActiveFunctions;

typedef enum KRBPNLearningModes
{
    //一般學習模式 (Normally), without any optimization algorithm (includes QuickProp).
    KRBPNLearningModeByNormal = 0,
    //智能混合學習速率 (Best), Input Fixed, Output Dynamic
    KRBPNLearningModeByQuickPropSmartHybrid,
    //固定慣性學習速率 (Better), Input & Output both Fixed
    KRBPNLearningModeByQuickPropFixed,
    //動態慣性學習速率 (Soso), Input & Output both Dynamic
    KRBPNLearningModeByQuickPropDynamic,
    //自訂固定慣性學習速率 (Better), Input & Output both custom Fixed
    KRBPNLearningModeByQuickPropCustomFixed
}KRBPNLearningModes;

typedef enum KRBPNEarlyStoppings
{
    //MSE
    KRBPNEarlyStoppingByMSE = 0,
    //RMSE
    KRBPNEarlyStoppingByRMSE,
}KRBPNEarlyStoppings;
