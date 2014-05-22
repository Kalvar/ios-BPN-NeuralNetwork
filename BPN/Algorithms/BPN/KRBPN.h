//
//  KRBPN.h
//  BPN ( 倒傳遞類神經網路 ) V0.1 Alpha
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2014年 Kuo-Ming Lin. All rights reserved.
//
/*
 * @ 3 層架構
 *   - 輸入層
 *   - 隱藏層
 *   - 輸出層
 */
#import <Foundation/Foundation.h>

/*
 * @ 儲存訓練過後的 BPN Network 數據資料
 */
#import "KRBPNTrainedNetwork.h"

/*
 * @ 訓練完成時
 *   - success     : 是否訓練成功
 *   - trainedInfo : 訓練後的 Network 資料
 */
typedef void(^KRBPNTrainingCompletion)(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes);

/*
 * @ 每一次的迭代資料
 *   - times       : 訓練到了第幾代
 *   - trainedInfo : 本次訓練的 Network 資料
 */
typedef void(^KRBPNEachGeneration)(NSInteger times, NSDictionary *trainedInfo);

/*
 * @ 當前訓練的 BPN Network 數據資料
 *   - trainedInfo = @{};
 *      - KRBPNTrainedInfoInputWeights      : NSMutableArray, 調整後的輸入層各向量值到隱藏層神經元的權重
 *      - KRBPNTrainedInfoHiddenWeights     : NSMutableArray, 調整後的隱藏層神經元到輸出層神經元的權重值
 *      - KRBPNTrainedInfoHiddenBiases      : NSMutableArray, 調整後的隱藏層神經元的偏權值
 *      - KRBPNTrainedInfoOutputBias        : double,         調整後的輸出層神經元偏權值
 *      - KRBPNTrainedInfoTrainedGeneration : NSInteger,      已訓練到第幾代
 */
static NSString *KRBPNTrainedInfoInputWeights      = @"KRBPNTrainedInfoInputWeights";
static NSString *KRBPNTrainedInfoHiddenWeights     = @"KRBPNTrainedInfoHiddenWeights";
static NSString *KRBPNTrainedInfoHiddenBiases      = @"KRBPNTrainedInfoHiddenBiases";
static NSString *KRBPNTrainedInfoOutputBias        = @"KRBPNTrainedInfoOutputBias";
static NSString *KRBPNTrainedInfoTrainedGeneration = @"KRBPNTrainedInfoTrainedGeneration";

@interface KRBPN : NSObject
{
    
}

//輸入層各向量值之陣列集合
@property (nonatomic, strong) NSMutableArray *inputs;
//輸入層各向量值到隱藏層神經元的權重
@property (nonatomic, strong) NSMutableArray *inputWeights;
//隱藏層神經元到輸出層神經元的權重值
@property (nonatomic, strong) NSMutableArray *hiddenWeights;
//隱藏層神經元的偏權值
@property (nonatomic, strong) NSMutableArray *hiddenBiases;
//隱藏層有幾顆神經元
@property (nonatomic, assign) NSInteger countHiddens;
//輸出層神經元偏權值
@property (nonatomic, assign) double outputBias;
//期望值
@property (nonatomic, assign) double targetValue;
//學習速率
@property (nonatomic, assign) CGFloat learningRate;
//收斂誤差值 ( 10^-3, 10^-6 )
@property (nonatomic, assign) double convergenceError;
//活化函式的 Alpha Value ( LMS 的坡度 )
@property (nonatomic, assign) float fOfAlpha;
//訓練迭代次數上限
@property (nonatomic, assign) NSInteger limitGeneration;
//目前訓練到第幾代
@property (nonatomic, assign) NSInteger trainingGeneration;
//是否正在訓練中
@property (nonatomic, assign) BOOL isTraining;
//當前訓練後的資料
@property (nonatomic, strong) NSDictionary *trainedInfo;
//取出儲存在 NSUserDefaults 裡訓練後的完整 BPN Network 數據資料
@property (nonatomic, readwrite) KRBPNTrainedNetwork *trainedNetwork;

@property (nonatomic, copy) KRBPNTrainingCompletion trainingCompletion;
@property (nonatomic, copy) KRBPNEachGeneration eachGeneration;

+(instancetype)sharedNetwork;
-(instancetype)init;

#pragma --mark Training Public Methods
-(void)training;
-(void)trainingDoneSave;
-(void)pause;
-(void)continueTraining;
-(void)reset;

#pragma --mark Trained Network Public Methods
-(void)saveTrainedNetwork;
-(void)removeTrainedNetwork;
-(void)recoverTrainedNetwork:(KRBPNTrainedNetwork *)_recoverNetworks;
-(void)recoverTrainedNetwork;

#pragma --mark Blocks
-(void)setTrainingCompletion:(KRBPNTrainingCompletion)_theBlock;
-(void)setEachGeneration:(KRBPNEachGeneration)_theBlock;

@end
