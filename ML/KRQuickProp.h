//
//  KRQuickProp.h
//  BPN
//
//  Created by Kalvar on 2015/5/16.
//  Copyright (c) 2015年 Kuo-Ming Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

typedef enum KRQuickPropLearningModes
{
    KRQuickPropLearningModeByInputFixed = 0,
    KRQuickPropLearningModeByInputDynamic,
    KRQuickPropLearningModeByOutputFixed,
    KRQuickPropLearningModeByOutputDynamic
    
}KRQuickPropLearningModes;

@interface KRQuickProp : NSObject

//Current Learning Rate
@property (nonatomic, assign) float outputLearningRate;
@property (nonatomic, assign) float inputLearningRate;

@property (nonatomic, assign) KRQuickPropLearningModes outputLearningMode;
@property (nonatomic, assign) KRQuickPropLearningModes inputLearningMode;

@property (nonatomic, strong) NSMutableArray *outputErrors;
@property (nonatomic, strong) NSMutableArray *outputResults;

@property (nonatomic, strong) NSMutableArray *hiddenErrors;
@property (nonatomic, strong) NSMutableArray *hiddenOutputs;
@property (nonatomic, strong) NSMutableArray *hiddenDeltaWeights; //隱藏層到輸出層的權重調節量

@property (nonatomic, strong) NSMutableArray *inputs;
@property (nonatomic, strong) NSMutableArray *inputDeltaWeights;  //輸入層到隱藏層的權重調節量

@property (nonatomic, assign) NSInteger times;
@property (nonatomic, assign) NSInteger patternIndex;             //目前正在使用哪一個 Input Pattern

+(instancetype)sharedInstance;
-(instancetype)init;

-(void)addOutputErrors:(NSArray *)_errors;
-(void)addOutputResults:(NSArray *)_outputs;
-(void)addHiddenErrors:(NSArray *)_errors;
-(void)addHiddenOutputs:(NSArray *)_outputs;
-(void)addHiddenDeltaWeights:(NSArray *)_weights;
-(void)addInputs:(NSArray *)_patterns;
-(void)addInputDeltaWeights:(NSArray *)_weights;

-(float)getHiddenDeltaWeightAtNetIndex:(NSInteger)_netIndex outputIndex:(NSInteger)_outputIndex;
-(float)getInputDeltaWeightAtNetIndex:(NSInteger)_netIndex outputIndex:(NSInteger)_outputIndex;

-(void)saveHiddenDeltaWeightAtNetIndex:(NSInteger)_netIndex outputIndex:(NSInteger)_outputIndex deltaWeight:(float)_weight;
-(void)saveInputDeltaWeightAtNetIndex:(NSInteger)_netIndex weightIndex:(NSInteger)_outputIndex deltaWeight:(float)_weight;

-(float)calculateOutputLearningRateAtNetIndex:(NSInteger)_netIndex outputIndex:(NSInteger)_outputIndex targetError:(float)_targetError hiddenOutput:(float)_hiddenOutput;
-(float)calculateInputLearningRateAtNetIndex:(NSInteger)_netIndex weightIndex:(NSInteger)_outputIndex hiddenError:(float)_targetError inputValue:(float)_inputValue;

-(void)clean;
-(void)plus;
-(void)reset;
-(void)setUsingPatternIndex:(NSInteger)_index;

-(void)setInputFixedRate:(float)_fixedRate;  //固定 Input 的學習速率
-(void)setOutputFixedRate:(float)_fixedRate; //固定 Output 的學習速率
-(void)setBothFixedRate:(float)_fixedRate;   //固定 Input & Output 的學習速率
-(void)setInputFixedRateByRandom;            //隨機在學術推薦範圍的 Input 固定學習速率
-(void)setOutputFixedRateByRandom;           //隨機在學術推薦範圍的 Output 固定學習速率
-(void)setBothFixedRateByRandom;             //隨機在學術推薦範圍的 Input & Output 固定學習速率

@end
