//
//  KRBPNTrainedNetwork.h
//  BPN V1.9.2
//
//  Created by Kalvar on 2014/5/22.
//  Copyright (c) 2014年 Kuo-Ming Lin (Kalvar). All rights reserved.
//

#import <Foundation/Foundation.h>

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
@property (nonatomic, assign) NSInteger limitGeneration;
@property (nonatomic, assign) NSInteger trainingGeneration;

+(instancetype)sharedNetwork;
-(instancetype)init;

@end
