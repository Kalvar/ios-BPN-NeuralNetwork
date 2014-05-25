//
//  KRBPNTrainedNetwork.h
//  BPN V1.1.7
//
//  Created by Kalvar on 2014/5/22.
//  Copyright (c) 2014年 Kuo-Ming Lin. All rights reserved.
//

@interface KRBPNTrainedNetwork : NSObject <NSCoding>

@property (nonatomic, strong) NSMutableArray *inputs;
@property (nonatomic, strong) NSMutableArray *inputWeights;
@property (nonatomic, strong) NSMutableArray *hiddenWeights;
@property (nonatomic, strong) NSMutableArray *hiddenBiases;
@property (nonatomic, assign) double outputBias;
@property (nonatomic, assign) NSArray *outputGoals;
@property (nonatomic, assign) CGFloat learningRate;
@property (nonatomic, assign) double convergenceError;
@property (nonatomic, assign) float fOfAlpha;
@property (nonatomic, assign) NSInteger limitGeneration;
@property (nonatomic, assign) NSInteger trainingGeneration;

+(instancetype)sharedNetwork;
-(instancetype)init;

@end
