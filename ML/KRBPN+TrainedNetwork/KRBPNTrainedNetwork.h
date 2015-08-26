//
//  KRBPNTrainedNetwork.h
//  BPN V2.0.1
//
//  Created by Kalvar on 2014/5/22.
//  Copyright (c) 2014年 Kuo-Ming Lin (Kalvar). All rights reserved.
//

#import "KRBPN+Definition.h"

@interface KRBPNTrainedNetwork : NSObject <NSCoding>

@property (nonatomic, strong) NSMutableArray *inputs;
@property (nonatomic, strong) NSMutableArray *inputWeights;
@property (nonatomic, strong) NSMutableArray *hiddenWeights;
@property (nonatomic, strong) NSMutableArray *hiddenBiases;
@property (nonatomic, strong) NSMutableArray *outputBiases;
@property (nonatomic, strong) NSArray *outputResults;
@property (nonatomic, strong) NSMutableArray *outputGoals;
@property (nonatomic, assign) CGFloat learningRate;
@property (nonatomic, assign) double convergenceError;
@property (nonatomic, assign) float fOfAlpha;
@property (nonatomic, assign) NSInteger limitIteration;
@property (nonatomic, assign) NSInteger presentIteration;
@property (nonatomic, assign) KRBPNActiveFunctions activeFunction;
@property (nonatomic, assign) KRBPNLearningModes learningMode;
@property (nonatomic, assign) KRBPNEarlyStoppings earlyStopping;
@property (nonatomic, strong) KRQuickProp *quickProp;

+(instancetype)sharedNetwork;
-(instancetype)init;

@end
