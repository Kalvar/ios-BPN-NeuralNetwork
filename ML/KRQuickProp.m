//
//  KRQuickProp.m
//  BPN
//
//  Created by Kalvar on 2015/5/16.
//  Copyright (c) 2015年 Kuo-Ming Lin. All rights reserved.
//
/*
 * This is not the completed implementation QuickProp theory (Fahlman), I used my experiences to modify a little part.
 * This method combined Kecman's advice that fixed learning rate in EDBD theory of neural network of enhancement, 0.5f ~ 0.7f is better.
 * Anyway, I believed never have any standard theory of neural network in any case.
 * 
 * 將 Fahlman 的建議從 1.75f 調到 0.75f, 或是都只使用 Kecman 的方法也都有不錯的表現效果。
 *
 */
#import "KRQuickProp.h"

#define QUICKPROP_MAX_DEFAULT_ADJUSTMENT_LEARNING_RATE   0.5f  //預設學習速率調節量
#define QUICKPROP_MAX_FAHLMAN_ADJUSTMENT_LEARNING_RATE   0.75f //最大動態學習速率調節量 ( Fahlman advised 1.75f, but I tested it better in 0.7f ~ 0.75f )
#define QUICKPROP_MAX_KECMAN_DEFAULT_FIXED_LEARNING_RATE 0.7f  //最大建議的固定學習速率 ( Kecman )
#define QUICKPROP_MIN_KECMAN_DEFAULT_FIXED_LEARNING_RATE 0.5f  //最小建議的固定學習速率 ( Kecman )

@interface KRQuickProp(fixQueue)

@property (nonatomic, strong) NSMutableArray *_deltaHiddenQueues;
@property (nonatomic, strong) NSMutableArray *_deltaInputQueues;

@end

@implementation KRQuickProp (ExtendMethods)

-(double)_randomMax:(double)_maxValue min:(double)_minValue
{
    return ((double)arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
}

-(float)_defaultFixedRandom
{
    return [self _randomMax:QUICKPROP_MAX_KECMAN_DEFAULT_FIXED_LEARNING_RATE
                        min:QUICKPROP_MIN_KECMAN_DEFAULT_FIXED_LEARNING_RATE];
}

//一開始針對上一次修正權重值的預設學習速率調節量
-(float)_defaultAdjustmentLearningRate
{
    return QUICKPROP_MAX_DEFAULT_ADJUSTMENT_LEARNING_RATE;
}

//最大 Fahlman 建議對上一次修正權重值的學習速率調節量為 1.75, 但自己的經驗是 0.7 ~ 0.75 最好
-(float)_fahlmanMaxAdjustmentLearningRate
{
    return QUICKPROP_MAX_FAHLMAN_ADJUSTMENT_LEARNING_RATE;
}

-(BOOL)_canAdjust
{
    return (self.times > 0);
}

@end

@implementation KRQuickProp

@synthesize outputLearningRate  = _outputLearningRate;
@synthesize inputLearningRate   = _inputLearningRate;

@synthesize outputLearningMode  = _outputLearningMode;
@synthesize inputLearningMode   = _inputLearningMode;

@synthesize outputErrors        = _outputErrors;
@synthesize outputResults       = _outputResults;

@synthesize hiddenErrors        = _hiddenErrors;
@synthesize hiddenOutputs       = _hiddenOutputs;
@synthesize hiddenDeltaWeights  = _hiddenDeltaWeights;

@synthesize inputs              = _inputs;
@synthesize inputDeltaWeights   = _inputDeltaWeights;

@synthesize times               = _times;
@synthesize patternIndex        = _patternIndex;

+(instancetype)sharedInstance
{
    static dispatch_once_t pred;
    static KRQuickProp *_object = nil;
    dispatch_once(&pred, ^
    {
        _object = [[KRQuickProp alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _outputLearningRate = 0.5f;
        _inputLearningRate  = 0.5f;
        
        _outputErrors       = [[NSMutableArray alloc] initWithCapacity:0];
        _outputResults      = [[NSMutableArray alloc] initWithCapacity:0];
        
        _hiddenErrors       = [[NSMutableArray alloc] initWithCapacity:0];
        _hiddenOutputs      = [[NSMutableArray alloc] initWithCapacity:0];
        _hiddenDeltaWeights = [[NSMutableArray alloc] initWithCapacity:0];
        
        _inputs             = [[NSMutableArray alloc] initWithCapacity:0];
        _inputDeltaWeights  = [[NSMutableArray alloc] initWithCapacity:0];
        
        _times              = 0;
        _patternIndex       = 0;
        
        _outputLearningMode = KRQuickPropLearningModeByOutputDynamic;
        _inputLearningMode  = KRQuickPropLearningModeByInputDynamic;
    }
    return self;
}

-(void)addOutputErrors:(NSArray *)_errors
{
    [_outputErrors addObjectsFromArray:_errors];
}

-(void)addOutputResults:(NSArray *)_outputs
{
    [_outputResults addObjectsFromArray:_outputs];
}

-(void)addHiddenErrors:(NSArray *)_errors
{
    [_hiddenErrors addObjectsFromArray:_errors];
}

-(void)addHiddenOutputs:(NSArray *)_outputs
{
    [_hiddenOutputs addObjectsFromArray:_outputs];
}

//增加調整後的 Hidden Layer 到 Output Layer 的權重值，如為初次新增，則 Network 初始權重即為其調整後的權重值
-(void)addHiddenDeltaWeights:(NSArray *)_weights
{
    [_hiddenDeltaWeights addObject:[[NSMutableArray alloc] initWithArray:_weights]];
}

-(void)addInputs:(NSArray *)_patterns
{
    [_inputs addObject:_patterns];
}

//增加調整後的 Input Layer 到 Hidden Layer 的權重值，如為初次新增，則 Network 初始權重即為其調整後的權重值
-(void)addInputDeltaWeights:(NSArray *)_weights
{
    //把 NSArray 轉成 NSMutableArray, 便於之後針對該權重位置的抽換
    [_inputDeltaWeights addObject:[[NSMutableArray alloc] initWithArray:_weights]];
}

//取出 Hidden Layer 調整後的權重值
-(float)getHiddenDeltaWeightAtNetIndex:(NSInteger)_netIndex outputIndex:(NSInteger)_outputIndex
{
    return [self _canAdjust] ? [[[_hiddenDeltaWeights objectAtIndex:_netIndex] objectAtIndex:_outputIndex] floatValue] : 0.0f;
}

//取出 Input Layer 調整後的權重值
-(float)getInputDeltaWeightAtNetIndex:(NSInteger)_netIndex outputIndex:(NSInteger)_outputIndex
{
    return [self _canAdjust] ? [[[_inputDeltaWeights objectAtIndex:_netIndex] objectAtIndex:_outputIndex] floatValue] : 0.0f;
}

/*
 * @ 儲存 Hidden Layer to Output Layer 調整後的權重值
 *   - _netIndex    : Which of Hidden Net
 *   - _outputIndex : Hidden Net 對應到哪一個 Output Net 的 Weight
 *   - _deltaWeight : Adjusted Weight Value or Initial Weight Value
 */
-(void)saveHiddenDeltaWeightAtNetIndex:(NSInteger)_netIndex outputIndex:(NSInteger)_outputIndex deltaWeight:(float)_deltaWeight
{
    [[_hiddenDeltaWeights objectAtIndex:_netIndex] replaceObjectAtIndex:_outputIndex
                                                             withObject:[NSNumber numberWithFloat:_deltaWeight]];
}

/*
 * @ 儲存 Input Layer to Hidden Layer 調整後的權重值
 *   - _netIndex    : Which of Input Net
 *   - _weightIndex : Input Net 對應到哪一個 Hidden Net 的 Weight
 *   - _weight      : Adjusted Weight Value or Initial Weight Value
 */
-(void)saveInputDeltaWeightAtNetIndex:(NSInteger)_netIndex weightIndex:(NSInteger)_weightIndex deltaWeight:(float)_deltaWeight
{
    [[_inputDeltaWeights objectAtIndex:_netIndex] replaceObjectAtIndex:_weightIndex
                                                            withObject:[NSNumber numberWithFloat:_deltaWeight]];
}

// Output Learning Rate
-(float)calculateOutputLearningRateAtNetIndex:(NSInteger)_netIndex outputIndex:(NSInteger)_outputIndex targetError:(float)_targetError hiddenOutput:(float)_hiddenOutput
{
    if( _outputLearningMode == KRQuickPropLearningModeByOutputFixed )
    {
        return _outputLearningRate;
    }
    
    if( ![self _canAdjust] )
    {
        return [self _defaultAdjustmentLearningRate];
    }
    
    float _lastTargetError  = [[_outputErrors objectAtIndex:_outputIndex] floatValue];
    float _lastHiddenOutput = [[_hiddenOutputs objectAtIndex:_netIndex] floatValue];
    float _learningRate     = 0.0f;
    float _lastAdjustedRate = -(_lastTargetError * _lastHiddenOutput);
    float _nowAdjustedRate  = -(_targetError * _hiddenOutput);
    
    // To avoid the large adjusted rate via Fahlman advised.
    if( _nowAdjustedRate > _lastAdjustedRate )
    {
        _learningRate = [self _fahlmanMaxAdjustmentLearningRate];
    }
    else
    {
        if( (_lastAdjustedRate - _nowAdjustedRate) != 0.0f )
        {
            // QuickProp, the adjusted Learning Rate
            _learningRate = _nowAdjustedRate / (_lastAdjustedRate - _nowAdjustedRate);
            if( _learningRate > [self _fahlmanMaxAdjustmentLearningRate] )
            {
                _learningRate = [self _fahlmanMaxAdjustmentLearningRate];
            }
        }
        else
        {
            _learningRate = [self _defaultAdjustmentLearningRate];
        }
    }
    
    //NSLog(@"Output LearningRate : %f", _learningRate);
    
    //is Nan, 分母為 0 時發生
    //if( _learningRate != _learningRate )
    
    return _outputLearningRate = _learningRate;
}

// Input Learning Rate
-(float)calculateInputLearningRateAtNetIndex:(NSInteger)_netIndex weightIndex:(NSInteger)_weightIndex hiddenError:(float)_hiddenError inputValue:(float)_inputValue
{
    if( _inputLearningMode == KRQuickPropLearningModeByInputFixed )
    {
        return _inputLearningRate;
    }
    
    if( ![self _canAdjust] )
    {
        return [self _defaultAdjustmentLearningRate];
    }
    
    float _lastHiddenError    = [[_hiddenErrors objectAtIndex:_weightIndex] floatValue];
    float _originalInputValue = [[[_inputs objectAtIndex:_patternIndex] objectAtIndex:_netIndex] floatValue];
    float _learningRate       = 0.0f;
    float _lastAdjustedRate   = -(_hiddenError * _inputValue);
    float _nowAdjustedRate    = -(_lastHiddenError * _originalInputValue);
    if( _nowAdjustedRate > _lastAdjustedRate )
    {
        _learningRate = [self _fahlmanMaxAdjustmentLearningRate];
    }
    else
    {
        if( (_lastAdjustedRate - _nowAdjustedRate) != 0.0f )
        {
            // QuickProp, the adjusted Learning Rate
            _learningRate = _nowAdjustedRate / (_lastAdjustedRate - _nowAdjustedRate);
            if( _learningRate > [self _fahlmanMaxAdjustmentLearningRate] )
            {
                _learningRate = [self _fahlmanMaxAdjustmentLearningRate];
            }
        }
        else
        {
            _learningRate = [self _defaultAdjustmentLearningRate];
        }
    }
    
    //NSLog(@"Input LearningRate : %f", _learningRate);
    
    return _inputLearningRate = _learningRate;
}

-(void)clean
{
    if( _times > 0 )
    {
        [_outputErrors removeAllObjects];
        [_outputResults removeAllObjects];
        
        [_hiddenErrors removeAllObjects];
        [_hiddenOutputs removeAllObjects];
        
        //[_hiddenDeltaWeights removeAllObjects];
        //[_inputs removeAllObjects];
        //[_inputDeltaWeights removeAllObjects];
    }
}

-(void)plus
{
    ++_times;
}

-(void)reset
{
    [self clean];
    _times = 0;
}

-(void)setUsingPatternIndex:(NSInteger)_index
{
    _patternIndex = _index;
}

#pragma --mark Setters
-(void)setInputFixedRate:(float)_fixedRate
{
    _inputLearningRate = _fixedRate;
    _inputLearningMode = KRQuickPropLearningModeByInputFixed;
}

-(void)setOutputFixedRate:(float)_fixedRate
{
    _outputLearningRate = _fixedRate;
    _outputLearningMode = KRQuickPropLearningModeByOutputFixed;
}

-(void)setBothFixedRate:(float)_fixedRate
{
    [self setInputFixedRate:_fixedRate];
    [self setOutputFixedRate:_fixedRate];
}

-(void)setInputFixedRateByRandom
{
    [self setInputFixedRate:[self _defaultFixedRandom]];
}

-(void)setOutputFixedRateByRandom
{
    [self setOutputFixedRate:[self _defaultFixedRandom]];
}

-(void)setBothFixedRateByRandom
{
    [self setBothFixedRate:[self _defaultFixedRandom]];
}

@end