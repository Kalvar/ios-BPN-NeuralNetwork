//
//  KRBPN.m
//  BPN V2.0.4
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2015年 Kuo-Ming Lin (Kalvar Lin). All rights reserved.
//

#import "KRBPN.h"

static NSString *_kOriginalInputs             = @"_kOriginalInputs";
static NSString *_kOriginalInputWeights       = @"_kOriginalInputWeights";
static NSString *_kOriginalHiddenWeights      = @"_kOriginalHiddenWeights";
static NSString *_kOriginalHiddenBiases       = @"_kOriginalHiddenBiases";
static NSString *_kOriginalOutputBiases       = @"_kOriginalOutputBiases";
static NSString *_kOriginalOutputResults      = @"_kOriginalOutputResults";
static NSString *_kOriginalOutputGoals        = @"_kOriginalOutputGoals";
static NSString *_kOriginalLearningRate       = @"_kOriginalLearningRate";
static NSString *_kOriginalConvergenceError   = @"_kOriginalConvergenceError";
static NSString *_kOriginalFOfAlpha           = @"_kOriginalFOfAlpha";
static NSString *_kOriginalLimitIterations    = @"_kOriginalLimitIterations";
static NSString *_kOriginalActiveFunction     = @"_kOriginalActiveFunction";
static NSString *_kOriginalLearningMode       = @"_kOriginalLearningMode";
static NSString *_kOriginalEarlyStopping      = @"_kOriginalEarlyStopping";
static NSString *_kOriginalQuickPropFixedRate = @"_kOriginalQuickPropFixedRate";
//static NSString *_kOriginalMaxMultiple      = @"_kOriginalMaxMultiple";

static NSString *_kTrainedNetworkInfo         = @"kTrainedNetworkInfo";

@interface KRBPN ()

//隱藏層的輸出值
@property (nonatomic, strong) NSArray *_hiddenOutputs;
//當前資料的輸出期望值
@property (nonatomic, assign) NSArray *_goalValues;
//輸出層的誤差值
@property (nonatomic, strong) NSArray *_outputErrors;
//是否強制中止訓練
@property (nonatomic, assign) BOOL _forceStop;
//原來的設定值
@property (nonatomic, strong) NSMutableDictionary *_originalParameters;
//訓練完就儲存至 NSUserDefaults 裡
@property (nonatomic, assign) BOOL _isDoneSave;
//記錄當前訓練到哪一組 Input 數據
@property (nonatomic, assign) NSInteger _patternIndex;
//在訓練 goalValue 且其值不在 0.0f ~ 1.0f 之間時，就使用本值進行相除與回乘原同類型值的動作
@property (nonatomic, assign) NSInteger _maxMultiple;
//儲存要用於比較計算 _maxMultiple 值的所有 Target Outputs
@property (nonatomic, strong) NSMutableArray *_compareTargets;
//儲存每一次 訓練迭代裡的每一個輸出誤差值, for MSE / RMSE
@property (nonatomic, strong) NSMutableArray *_patternErrors;
//儲存每一個迭代的誤差總和
//@property (nonatomic, strong) NSMutableArray *_iterationErrors;

@property (nonatomic, strong) KRQuickProp *_quickProp;

@end

@implementation KRBPN (fixInitials)

-(void)_resetTrainedParameters
{
    self.outputResults       = nil;
    
    self.trainedNetwork      = nil;
    self.presentIteration    = 0;
    
    self._hiddenOutputs      = nil;
    self._outputErrors       = nil;
    self._forceStop          = NO;
    self._isDoneSave         = NO;
    self._patternIndex       = 0;
}

-(void)_initWithVars
{
    self.delegate            = nil;
    self.inputs              = [[NSMutableArray alloc] initWithCapacity:0];
    self.inputWeights        = [[NSMutableArray alloc] initWithCapacity:0];
    self.hiddenWeights       = [[NSMutableArray alloc] initWithCapacity:0];
    self.hiddenBiases        = [[NSMutableArray alloc] initWithCapacity:0];
    self.countHiddenNets     = 0;
    self.outputBiases        = [NSMutableArray new];
    self.outputGoals         = [NSMutableArray new];
    self.learningRate        = 0.8f;
    self.convergenceError    = 0.001f;
    self.fOfAlpha            = 1;
    self.limitIteration      = 0;
    self.isTraining          = NO;
    self.trainedInfo         = nil;
    
    self.trainingCompletion  = nil;
    self.eachIteration       = nil;
    
    self._quickProp          = [[KRQuickProp alloc] init]; //[KRQuickProp sharedInstance];
    
    [self _resetTrainedParameters];
    
    self._maxMultiple        = 1;
    self._goalValues         = nil;
    self._originalParameters = [NSMutableDictionary new];
    self._compareTargets     = [NSMutableArray new];
    self._patternErrors      = [NSMutableArray new];
    //self._iterationErrors  = [NSMutableArray new];
    
    self.activeFunction      = KRBPNActivationBySigmoid;
    self.learningMode        = KRBPNLearningModeByNormal;
    self.earlyStopping       = KRBPNEarlyStoppingByMSE;
    self.quickPropFixedRate  = 0.0f;
    
}

@end

@implementation KRBPN (fixMethods)

-(void)_stopTraining
{
    self.isTraining = NO;
}

-(void)_completedTraining
{
    self.isTraining  = NO;
    if( self._isDoneSave )
    {
        self._isDoneSave = NO;
        [self saveNetwork];
    }
    
    if( self.delegate )
    {
        if( [self.delegate respondsToSelector:@selector(krBpnDidTrainFinished:trainedInfo:totalTimes:)] )
        {
            [self.delegate krBpnDidTrainFinished:self trainedInfo:self.trainedInfo totalTimes:self.presentIteration];
        }
    }
    
    if( self.trainingCompletion )
    {
        self.trainingCompletion(YES, self.trainedInfo, self.presentIteration);
    }
}

-(void)_printEachIteration
{
    if( self.delegate )
    {
        if( [self.delegate respondsToSelector:@selector(krBpnEachIteration:trainedInfo:times:)] )
        {
            [self.delegate krBpnEachIteration:self trainedInfo:self.trainedInfo times:self.presentIteration];
        }
    }
    
    if( self.eachIteration )
    {
        self.eachIteration(self.presentIteration, self.trainedInfo);
    }
}

-(void)_copyParameters
{
    if( !self._originalParameters )
    {
        self._originalParameters = [NSMutableDictionary new];
    }
    else
    {
        [self._originalParameters removeAllObjects];
    }
    // Its memory references & links the _originalParameters.
    NSMutableDictionary *_originals = self._originalParameters;
    [_originals setObject:[self.inputs copy] forKey:_kOriginalInputs];
    [_originals setObject:[self.inputWeights copy] forKey:_kOriginalInputWeights];
    [_originals setObject:[self.hiddenWeights copy] forKey:_kOriginalHiddenWeights];
    [_originals setObject:[self.hiddenBiases copy] forKey:_kOriginalHiddenBiases];
    [_originals setObject:[self.outputBiases copy] forKey:_kOriginalOutputBiases];
    [_originals setObject:[self.outputGoals copy] forKey:_kOriginalOutputGoals];
    [_originals setObject:[NSNumber numberWithFloat:self.learningRate] forKey:_kOriginalLearningRate];
    [_originals setObject:[NSNumber numberWithDouble:self.convergenceError] forKey:_kOriginalConvergenceError];
    [_originals setObject:[NSNumber numberWithFloat:self.fOfAlpha] forKey:_kOriginalFOfAlpha];
    [_originals setObject:[NSNumber numberWithInteger:self.limitIteration] forKey:_kOriginalLimitIterations];
    [_originals setObject:[NSNumber numberWithInt:self.activeFunction] forKey:_kOriginalActiveFunction];
    [_originals setObject:[NSNumber numberWithInt:self.learningMode] forKey:_kOriginalLearningMode];
    [_originals setObject:[NSNumber numberWithInt:self.earlyStopping] forKey:_kOriginalEarlyStopping];
    [_originals setObject:[NSNumber numberWithFloat:self.quickPropFixedRate] forKey:_kOriginalQuickPropFixedRate];
    //[_originals setObject:[NSNumber numberWithInteger:self._maxMultiple] forKey:_kOriginalMaxMultiple];
}

-(void)_recoverOriginalParameters
{
    NSMutableDictionary *_originals = self._originalParameters;
    if( _originals )
    {
        if( [_originals count] > 0 )
        {
            [self.inputs removeAllObjects];
            [self.inputs addObjectsFromArray:[_originals objectForKey:_kOriginalInputs]];
            
            [self.inputWeights removeAllObjects];
            [self.inputWeights addObjectsFromArray:[_originals objectForKey:_kOriginalInputWeights]];
            
            [self.hiddenWeights removeAllObjects];
            [self.hiddenWeights addObjectsFromArray:[_originals objectForKey:_kOriginalHiddenWeights]];
            
            [self.hiddenBiases removeAllObjects];
            [self.hiddenBiases addObjectsFromArray:[_originals objectForKey:_kOriginalHiddenBiases]];
            
            self.outputBiases       = (NSMutableArray *)[_originals objectForKey:_kOriginalOutputBiases];
            self.outputGoals        = (NSMutableArray *)[_originals objectForKey:_kOriginalOutputGoals];
            self.learningRate       = [[_originals objectForKey:_kOriginalLearningRate] floatValue];
            self.convergenceError   = [[_originals objectForKey:_kOriginalConvergenceError] doubleValue];
            self.fOfAlpha           = [[_originals objectForKey:_kOriginalFOfAlpha] floatValue];
            self.limitIteration     = [[_originals objectForKey:_kOriginalLimitIterations] integerValue];
            
            self.activeFunction     = (KRBPNActiveFunctions)[[_originals objectForKey:_kOriginalActiveFunction] intValue];
            self.learningMode       = (KRBPNLearningModes)[[_originals objectForKey:_kOriginalLearningMode] intValue];
            self.earlyStopping      = (KRBPNEarlyStoppings)[[_originals objectForKey:_kOriginalEarlyStopping] intValue];
            self.quickPropFixedRate = [[_originals objectForKey:_kOriginalQuickPropFixedRate] floatValue];
            
            //self._maxMultiple     = [[_originals objectForKey:_kOriginalMaxMultiple] integerValue];
        }
    }
}

/*
 * @ 亂數給指定範圍內的值
 *   - ex : 1.0 ~ -1.0
 */
-(double)_randomMax:(double)_maxValue min:(double)_minValue
{
    /*
     * @ 2014.12.28 PM 20:15
     * @ Noted
     *   - rand() not fits to use here.
     *   - arc4random() fits here, it's the real random number.
     *
     * @ Samples
     *   - srand((int)time(NULL));
     *     double _random = ((double)rand() / RAND_MAX) * (_maxValue - _minValue) + _minValue;
     *     RAND_MAX 是 0x7fffffff (2147483647)，而 arc4random() 返回的最大值则是 0x100000000 (4294967296)，
     *     故 * 2.0f 待除，或使用自訂義 ARC4RANDOM_MAX      0x100000000
     */
    return ((double)arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
}

@end

@implementation KRBPN (fixQuickProp)

-(void)_addInputs:(NSArray *)_inputPatterns
{
    [self._quickProp addInputs:_inputPatterns];
}

-(void)_addInputDeltaWeights:(NSArray *)_weights
{
    [self._quickProp addInputDeltaWeights:_weights];
}

-(void)_addHiddenDeltaWeights:(NSArray *)_weights
{
    [self._quickProp addHiddenDeltaWeights:_weights];
}

-(BOOL)_isUsingQuickProp
{
    return ( self.learningMode != KRBPNLearningModeByNormal );
}

@end

@implementation KRBPN (fixFOfNets)
/*
 * @ S 形函數
 *   - [0.0, 1.0]
 */
-(float)_fOfSigmoid:(float)_x
{
    return ( 1 / ( 1 + powf(M_E, (-(self.fOfAlpha) * _x)) ) );
}

/*
 * @ 雙曲線函數
 *   - [-1.0, 1.0]
 */
-(float)_fOfTanh:(float)_x
{
    return ( 2.0f / ( 1 + powf(M_E, (-2.0f * _x)) ) ) - 1.0f;
}

/*
 * @ Fuzzy function
 *   - Still waiting for implementing.
 */
-(float)_fOfFuzzy:(float)_x
{
    //Do Fuzzy ...
    return -0.1f;
}

-(float)_fOfX:(float)_x
{
    float _y = 0.0f;
    switch (self.activeFunction)
    {
        case KRBPNActivationByTanh:
            _y = [self _fOfTanh:_x];
            break;
        case KRBPNActivationByFuzzy:
            _y = [self _fOfFuzzy:_x];
            break;
        case KRBPNActivationBySigmoid:
            _y = [self _fOfSigmoid:_x];
        default:
            break;
    }
    //isNaN ( not a number )
    /*
    if( _y != _y )
    {
        [self restart];
    }
     */
    return _y;
}

@end

@implementation KRBPN (fixTrainings)
/*
 * @ 計算隱藏層各個神經元( Nets )的輸出值
 *
 *   - 1. 累加神經元的值 SUM(net)
 *     - net(j) = ( SUM W(ji) * X(i) ) + b(j)
 *
 *   - 2. 代入活化函式 f(net), LMS 最小均方法
 *
 */
-(NSArray *)_sumHiddenLayerOutputsFromInputs:(NSArray *)_inputs
{
    //運算完成的 Nets = net(j)
    NSMutableArray *_fOfNets = [NSMutableArray new];
    //輸入層要做 SUM 就必須取出同一維度的值做運算
    //定義權重的維度
    int _weightDimesion      = -1;
    //依照隱藏層有幾顆神經元(偏權值)，就跑幾次的維度運算
    for( NSNumber *_partialWeight in self.hiddenBiases )
    {
        ++_weightDimesion;
        //再以同維度做 SUM 方法
        float _sumOfNet = 0;
        //有幾個 Input 就有幾個 Weight
        //取出每一個輸入值( Ex : X1 轉置矩陣後的輸入向量 [1, 2, -1] )
        int _inputIndex = -1;
        for( NSNumber *_xi in _inputs )
        {
            ++_inputIndex;
            //取出每一個同維度的輸入層到隱藏層的權重
            NSArray *_everyWeights = [self.inputWeights objectAtIndex:_inputIndex];
            //將值與權重相乘後累加，例如 : SUM( w14 x 0.2 + w24 x 0.4 + w34 x -0.5 ), SUM( w15 x -0.3 + w25 x 0.1 + w35 x 0.2 ) ...
            float _weight  = [[_everyWeights objectAtIndex:_weightDimesion] floatValue];
            _sumOfNet     += [_xi floatValue] * _weight;
            //NSLog(@"xValue : %f, _weight : %f", [_xi floatValue], _weight);
        }
        /*
         * @ 隱藏層神經元的偏權值
         *
         *   - hiddenNetPartialWeights = @[@-0.4, @0.2]; //_partialWeight
         *
         * @ 作 SUM 加總的本意
         *
         *   - 要明白 SUM 的本意，是指「加總融合」所有輸入向量陣列裡的值 ( 做線性代數的一維矩陣相乘 )
         */
        //減同維度的神經元偏權值
        _sumOfNet    -= [_partialWeight floatValue];
        float _fOfNet = [self _fOfX:_sumOfNet];
        [_fOfNets addObject:[NSNumber numberWithFloat:_fOfNet]];
    }
    return ( [_fOfNets count] > 0 ) ? _fOfNets : nil;
}

/*
 * @ 計算輸出層神經元的值
 *   - 跟計算隱藏層 [self _sumHiddenLayerNetWeightsFromInputs:] 一樣的公式。
 */
-(NSArray *)_sumOutputLayerNetsValue
{
    NSMutableArray *_fOfNets = nil;
    NSArray *_hideOutputs    = self._hiddenOutputs;
    if( _hideOutputs )
    {
        _fOfNets             = [NSMutableArray new];
        int _outputIndex     = -1;
        for( NSNumber *_outputBias in self.outputBiases )
        {
            float _sumOfNet = 0;
            ++_outputIndex;
            int _netIndex   = -1;
            for( NSArray *_weights in self.hiddenWeights )
            {
                ++_netIndex;
                NSNumber *_outputWeight  = [_weights objectAtIndex:_outputIndex];
                _sumOfNet               += [[_hideOutputs objectAtIndex:_netIndex] floatValue] * [_outputWeight floatValue];
            }
            _sumOfNet        -= [_outputBias floatValue];
            float _netOutput  = [self _fOfX:_sumOfNet];
            [_fOfNets addObject:[NSNumber numberWithFloat:_netOutput]];
        }
    }
    return _fOfNets;
}

/*
 * @ 計算輸出層神經元( Net )的輸出誤差
 *   - 公式 : Oj x ( 1 - Oj ) x ( Tj - Oj )
 */
-(NSArray *)_calculateOutputError
{
    NSMutableArray *_errors = nil;
    self.outputResults      = [self _sumOutputLayerNetsValue];
    NSArray *_netOutputs    = self.outputResults;
    if( _netOutputs )
    {
        _errors = [NSMutableArray new];
        NSInteger _outputIndex = -1;
        for( NSNumber *_netOutput in _netOutputs )
        {
            ++_outputIndex;
            //取出輸出層神經元的輸出結果
            float _outputValue = [_netOutput floatValue];
            float _targetValue = [[self._goalValues objectAtIndex:_outputIndex] floatValue] / self._maxMultiple;
            //計算與期望值的誤差
            float _errorValue  = _outputValue * ( 1 - _outputValue ) * ( _targetValue - _outputValue );
            [_errors addObject:[NSNumber numberWithFloat:_errorValue]];
            //儲存每一個 Pattern 的輸出誤差 (Tj - Yj)
            float _mseError    = _targetValue - _outputValue;
            [self._patternErrors addObject:[NSNumber numberWithFloat:_mseError]];
        }
    }
    return _errors;
}

/*
 * @ 計算隱藏層神經元( Nets )的誤差
 *   - 公式 : Oj x ( 1 - Oj ) x Errork x Wjk
 */
-(NSArray *)_calculateNetsError
{
    NSMutableArray *_netErrors = nil;
    self._outputErrors         = [self _calculateOutputError];
    NSArray *_errors           = self._outputErrors;
    if( _errors )
    {
        //[self._iterationErrors addObject:[self._outputErrors copy]];
        _netErrors    = [NSMutableArray new];
        int _netIndex = -1;
        for( NSNumber *_output in self._hiddenOutputs )
        {
            ++_netIndex;
            float _hiddenOutput = [_output floatValue];
            int _weightIndex    = -1;
            float _sumError     = 0.0f;
            //SUM output layer errors
            for( NSNumber *_outputError in _errors )
            {
                ++_weightIndex;
                float _netWeight    = [[[self.hiddenWeights objectAtIndex:_netIndex] objectAtIndex:_weightIndex] floatValue];
                float _hiddenError  = [_outputError floatValue] * _netWeight;
                _sumError          += _hiddenError;
            }
            //微分, S * Hidden layer net output * ( 1 - Hidden layer net output )
            _sumError *= _hiddenOutput * ( 1 - _hiddenOutput );
            [_netErrors addObject:[NSNumber numberWithFloat:_sumError]];
        }
    }
    return _netErrors;
}

/*
 * @ 更新權重與偏權值
 *   - 公式 : Shita(i) = Shita(i) + learning rate x Error(k)
 *              偏權值 = 偏權值    +   學習速率      x 要修改的誤差值
 */
-(BOOL)_refreshNetsWeights
{
    if( self._forceStop )
    {
        [self _stopTraining];
        return NO;
    }
    
    self.isTraining         = YES;
    BOOL _onePatternTrained = NO;
    //隱藏層神經元的輸出誤差值
    NSArray *_hiddenErrors  = [self _calculateNetsError];
    if( _hiddenErrors )
    {
        //更新隱藏層與輸出層的偏權值和權重
        NSMutableArray *_updatedOutputBiases  = [NSMutableArray new];
        NSMutableArray *_updatedHiddenWeights = [NSMutableArray new];
        NSMutableArray *_updatedHiddenBiases  = [NSMutableArray new];
        int _outputIndex                      = -1;
        for( NSNumber *_outputError in self._outputErrors )
        {
            ++_outputIndex;
            //更新輪出層偏權值
            float _targetError             = [_outputError floatValue];
            float _outputBias              = [[self.outputBiases objectAtIndex:_outputIndex] floatValue];
            _outputBias                   += ( -self.learningRate * _targetError );
            [_updatedOutputBiases addObject:[NSNumber numberWithFloat:_outputBias]];
            
            //更新隱藏層到輸出層的所有權重
            NSMutableArray *_unsortWeights = [NSMutableArray new];
            int _netIndex                  = -1;
            //取出指定神經元的所有權重值
            NSArray *_weights              = [self.hiddenWeights copy];
            for( NSArray *_netWeights in _weights )
            {
                ++_netIndex;
                //ex : W11, W21, W31
                float _hiddenOutput  = [[self._hiddenOutputs objectAtIndex:_netIndex] floatValue];
                float _weight        = [[_netWeights objectAtIndex:_outputIndex] floatValue];
                float _deltaWeight   = ( self.learningRate * _targetError * _hiddenOutput );
                
                if( [self _isUsingQuickProp] )
                {
                    float _quickpropLearningRate = [self._quickProp calculateOutputLearningRateAtNetIndex:_netIndex
                                                                                                outputIndex:_outputIndex
                                                                                                targetError:_targetError
                                                                                               hiddenOutput:_hiddenOutput];;
                    float _lastDeltaWeight       = [self._quickProp getHiddenDeltaWeightAtNetIndex:_netIndex outputIndex:_outputIndex];
                    _deltaWeight                += ( _quickpropLearningRate * _lastDeltaWeight );
                    [self._quickProp saveHiddenDeltaWeightAtNetIndex:_netIndex outputIndex:_outputIndex deltaWeight:_deltaWeight];
                }
                
                // + Delta Weight (權重調整量)
                _weight += _deltaWeight;
                
                //將連接到同一個 Output Net 的權重值作修正後的儲存，例如 : 連接到 Net8 的有 W58, W68, W78，都會放在同一維度陣列裡
                [_unsortWeights addObject:[NSNumber numberWithFloat:_weight]];
            }
            //將相對應 Hidden Layer Net 的權重放入, _unsortWeights = @[W58, W68, W78]、@[W59, W69, W79] ...
            [_updatedHiddenWeights addObject:_unsortWeights];
            //[self.hiddenWeights replaceObjectAtIndex:_hiddenIndex withObject:[NSNumber numberWithFloat:_weight]];
            _weights = nil;
        }
        [self.outputBiases removeAllObjects];
        [self.outputBiases addObjectsFromArray:_updatedOutputBiases];
        _updatedOutputBiases = nil;
        
        //更新隱藏層神經元的偏權值
        for( NSNumber *_hiddenError in _hiddenErrors )
        {
            float _netBias         = [[self.hiddenBiases objectAtIndex:_outputIndex] floatValue];
            // + Delta Bias (偏權值調整量)
            float _renewHiddenBias = _netBias + ( -self.learningRate * [_hiddenError floatValue] );
            [_updatedHiddenBiases addObject:[NSNumber numberWithFloat:_renewHiddenBias]];
        }
        [self.hiddenBiases removeAllObjects];
        [self.hiddenBiases addObjectsFromArray:_updatedHiddenBiases];
        _updatedHiddenBiases = nil;
        
        //將各個神經元的權重歸定位
        NSMutableArray *_sortedWeights = [NSMutableArray new];
        NSInteger _hiddenCount         = [self countHiddenNets];
        for( int _netIndex=0; _netIndex < _hiddenCount; _netIndex++ )
        {
            NSMutableArray *_sameWeights = [NSMutableArray new];
            for( NSArray *_weights in _updatedHiddenWeights )
            {
                NSNumber *_updatedWeight = [_weights objectAtIndex:_netIndex];
                [_sameWeights addObject:_updatedWeight];
            }
            [_sortedWeights addObject:_sameWeights];
        }
        
        [self.hiddenWeights removeAllObjects];
        [self.hiddenWeights addObjectsFromArray:_sortedWeights];
        _sortedWeights = nil;
        
        //最後更新所有輸入層到隱藏層的權重
        NSArray *_weights = [self.inputWeights copy];
        int _inputIndex   = -1;
        for( NSArray *_netWeights in _weights )
        {
            ++_inputIndex;
            int _weightIndex              = -1;
            NSMutableArray *_resetWeights = [NSMutableArray new];
            for( NSNumber *_everyWeight in _netWeights )
            {
                //每一個權重陣列的元素個數，會等於隱藏層神經元個數
                ++_weightIndex;
                float _netWeight   = [_everyWeight floatValue];
                float _hiddenError = [[_hiddenErrors objectAtIndex:_weightIndex] floatValue];
                float _inputValue  = [[[self.inputs objectAtIndex:self._patternIndex] objectAtIndex:_inputIndex] floatValue];
                float _resetWeight = 0.0f;
                float _deltaWeight = ( self.learningRate * _hiddenError * _inputValue );
                
                if( [self _isUsingQuickProp] )
                {
                    float _quickpropLearningRate = [self._quickProp calculateInputLearningRateAtNetIndex:_inputIndex
                                                                                               weightIndex:_weightIndex
                                                                                               hiddenError:_hiddenError
                                                                                                inputValue:_inputValue];
                    float _lastDeltaWeight = [self._quickProp getInputDeltaWeightAtNetIndex:_inputIndex outputIndex:_weightIndex];
                    _deltaWeight           +=( _quickpropLearningRate * _lastDeltaWeight );
                    _resetWeight           = _netWeight + _deltaWeight;
                    [self._quickProp saveInputDeltaWeightAtNetIndex:_inputIndex weightIndex:_weightIndex deltaWeight:_deltaWeight];
                }
                else
                {
                    _resetWeight = _netWeight + _deltaWeight;
                }
                
                [_resetWeights addObject:[NSNumber numberWithFloat:_resetWeight]];
                //NSLog(@"_new weight : %f = %f + ( %f * %f * %f )", _resetWeight, _netWeight, self.learningRate, _hiddenError, _inputValue);
            }
            //修正 InputWeights 輸入層到隱藏層的權重
            [self.inputWeights replaceObjectAtIndex:_inputIndex withObject:_resetWeights];
        }
        _weights           = nil;
        _onePatternTrained = YES;
        
        if( [self _isUsingQuickProp] )
        {
            [self._quickProp clean];
            [self._quickProp addOutputErrors:self._outputErrors];
            [self._quickProp addOutputResults:self.outputResults];
            [self._quickProp addHiddenErrors:_hiddenErrors];
            [self._quickProp addHiddenOutputs:self._hiddenOutputs];
            [self._quickProp plus];
        }
    }
    
    return _onePatternTrained;
}

//格式化訓練用的期望值至指定值域 [0.0, 1.0] 之間
-(void)_formatMaxMultiple
{
    //先找出期望值的最大絕對值
    NSNumber *_max  = [self._compareTargets valueForKeyPath:@"@max.self"];
    NSNumber *_min  = [self._compareTargets valueForKeyPath:@"@min.self"];
    double _fabsMax = fabs(_max.doubleValue);
    double _fabsMin = fabs(_min.doubleValue);
    double _realMax = MAX(_fabsMax, _fabsMin);
    if( _realMax > 1.0f )
    {
        self._maxMultiple  = 10 * ( (int)log10(_realMax) + 1 );
    }
}

//均方誤差
-(BOOL)_mse
{
    float _sumOutputError = 0.0f;
    for( NSNumber *_outpurError in self._patternErrors )
    {
        //使用絕對值來做誤差比較 error ^ 2
        _sumOutputError += ( [_outpurError floatValue] * [_outpurError floatValue] );
    }
    
    //MSE, 除以 (訓練範例數目 x 數出層神經元數目)
    //_sumOutputError = _sumOutputError / ([self.inputs count] * [self.outputBiases count]);
    _sumOutputError = _sumOutputError / [self._patternErrors count] * 0.5f;
    
    //如果已達收斂誤差，就不再繼續訓練
    return ( _sumOutputError <= self.convergenceError );
}

//均方根誤差
-(BOOL)_rmse
{
    float _sumOutputError = 0.0f;
    for( NSNumber *_outpurError in self._patternErrors )
    {
        //使用絕對值來做誤差比較 error ^ 2
        _sumOutputError += ( [_outpurError floatValue] * [_outpurError floatValue] );
    }
    
    //RMSE, 除以 (訓練範例數目 x 數出層神經元數目)
    //_sumOutputError = sqrtf(_sumOutputError / ([self.inputs count] * [self.outputBiases count]));
    _sumOutputError = sqrtf(_sumOutputError / [self._patternErrors count]);
    return ( _sumOutputError <= self.convergenceError );
}

-(void)_startTraining
{
    ++self.presentIteration;
    self._patternIndex = -1;
    /*
     * @ 依公式所說，X(i) 輸入向量應做轉置矩陣運算，但轉置矩陣須耗去多餘效能，
     *   因此，這裡暫不採用直接先轉成轉置矩陣的動作，
     *   而是直接依照資料長度取出同維度的方式來做轉置矩陣。
     *
     * @ 如輸入向量是 X1 = [1, 2, 3]; 的多值型態，就採用線性代數的解法，
     *   - 要將 X1 先轉置矩陣變成 :
     *             [1]
     *     X1(T) = [2]
     *             [3]
     *     這為第 1 筆訓練資料，當成輸入層神經元代入，此時輸入層就有 3 顆神經元。
     */
    //開始代入每一個訓練範例 X1, X2 ... Xn 各組的訓練資料
    for( NSArray *_inputs in self.inputs )
    {
        ++self._patternIndex;
        // Is using QuickProp ?
        if( [self _isUsingQuickProp] )
        {
            [self._quickProp setUsingPatternIndex:self._patternIndex];
        }
        /*
         * @ 每一筆輸入向量( 每組訓練的 Pattern )都會有自己的輸出期望值
         *   － 例如 : 
         *           輸入 X1[1, 0, 0]，其期望輸出為 1.0
         *           輸入 X2[0, 1, 0]，其期望輸出為 2.0
         *           輸入 X3[0, 0, 1]，其期望輸出為 3.0
         *      將輸出期望值正規化後，會變成 0.1, 0.2, 0.3
         *      以此類推。
         *
         * @ 不論正負號都先轉成絕對值，我只要求得除幾位數變成小數點
         *   - NSLog(@"%i", (int)log10f(-81355.555)); //-2147483648
         *   - NSLog(@"%i", (int)log10f(81355.555)); //4 個 10 倍
         *
         */
        self._goalValues    = [self.outputGoals objectAtIndex:self._patternIndex];
        self._hiddenOutputs = [self _sumHiddenLayerOutputsFromInputs:_inputs];
        //更新權重失敗，代表訓練異常，中止 !
        if ( ![self _refreshNetsWeights] )
        {
            //考慮，是否要記錄訓練到哪一筆，等等再繼續 ?
            //要繼續的話應該要重頭再來才是 ?
            break;
        }
    }
    
    //如有指定迭代數 && 當前訓練迭代數 >= 指定迭代數
    if( self.limitIteration > 0 && self.presentIteration >= self.limitIteration )
    {
        //停止訓練
        [self _completedTraining];
        return;
    }
    
    BOOL _isDone = ( self.earlyStopping == KRBPNEarlyStoppingByMSE ) ? [self _mse] : [self _rmse];
    [self._patternErrors removeAllObjects];
    //檢查是否收斂
    if( _isDone )
    {
        //達到收斂誤差值或出現異常狀況，即停止訓練
        [self _completedTraining];
        return;
    }
    else
    {
        //全部數據都訓練完了，才為 1 迭代
        [self _printEachIteration];
        //未達收斂誤差，則繼續執行訓練
        [self _startTraining];
    }
}

-(void)_trainingWithExtraHandler:(void(^)())_extraHandler
{
    //DISPATCH_QUEUE_CONCURRENT
    dispatch_queue_t queue = dispatch_queue_create("com.krbpn.train-network", NULL);
    dispatch_async(queue, ^(void)
    {
        [self pause];
        [self _resetTrainedParameters];
        if( _extraHandler )
        {
            _extraHandler();
        }
        [self _copyParameters];
        dispatch_async(dispatch_get_main_queue(), ^
        {
            [self _formatMaxMultiple];
            [self _startTraining];
        });
    });
}

@end

@implementation KRBPN

@synthesize delegate            = _delegate;

@synthesize inputs              = _inputs;
@synthesize inputWeights        = _inputWeights;
@synthesize hiddenWeights       = _hiddenWeights;
@synthesize hiddenBiases        = _hiddenBiases;
@synthesize countHiddenNets     = _countHiddenNets;
@synthesize outputBiases        = _outputBiases;
@synthesize outputResults       = _outputResults;
@synthesize outputGoals         = _outputGoals;
@synthesize learningRate        = _learningRate;
@synthesize convergenceError    = _convergenceError;
@synthesize fOfAlpha            = _fOfAlpha;
@synthesize limitIteration      = _limitIteration;
@synthesize presentIteration    = _presentIteration;
@synthesize isTraining          = _isTraining;
@synthesize trainedInfo         = _trainedInfo;
@synthesize trainedNetwork      = _trainedNetwork;
@synthesize activeFunction      = _activeFunction;
@synthesize trainingCompletion  = _trainingCompletion;
@synthesize eachIteration       = _eachIteration;
@synthesize learningMode        = _learningMode;
@synthesize earlyStopping       = _earlyStopping;
@synthesize quickPropFixedRate  = _quickPropFixedRate;

@synthesize _hiddenOutputs;
@synthesize _goalValues;
@synthesize _outputErrors;
@synthesize _forceStop;
@synthesize _originalParameters;
@synthesize _isDoneSave;
@synthesize _patternIndex;
@synthesize _maxMultiple;
@synthesize _compareTargets;
@synthesize _patternErrors;
//@synthesize _iterationErrors;

@synthesize _quickProp;

+(instancetype)sharedNetwork
{
    static dispatch_once_t pred;
    static KRBPN *_object = nil;
    dispatch_once(&pred, ^
    {
        _object = [[KRBPN alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        [self _initWithVars];
    }
    return self;
}

#pragma --mark Settings Public Methods
/*
 * @params, _patterns : 各輸入向量陣列值
 * @params, _weights  : 輸入層各向量值到隱藏層神經元的權重
 * @params, _goals    : 每一筆輸入向量的期望值( 輸出期望 )
 */
-(void)addPatterns:(NSArray *)_patterns outputGoals:(NSArray *)_goals
{
    [_inputs addObject:_patterns];
    [_outputGoals addObject:_goals]; //@[ @[1.0, 0.8, 0.2], @[0.76, 0.5, 0.89], ... ]
    [self._compareTargets addObjectsFromArray:_goals];
    [self _addInputs:_patterns];
}

/*
 * @ Each input data will match how many nets of hidden layer via setting the weights
 */
-(void)addPatternWeights:(NSArray *)_weights
{
    [_inputWeights addObject:_weights]; //@[ @[W14, W15], @[W24, W25], @[W34, W35] ]
    [self _addInputDeltaWeights:_weights];
}

/*
 * @ 增加隱藏層的各項參數設定
 *   - _netBias 隱藏層神經元 Net # 的偏權值
 *   - _netWeights 隱藏層神經元 Net # 到下一層神經元的權重值
 */
-(void)addHiddenLayerNetBias:(float)_netBias outputWeights:(NSArray *)_outputWeights
{
    [_hiddenBiases addObject:[NSNumber numberWithFloat:_netBias]];
    //@[ @[W57, W58, W59], @[W67, W68, W69], ... ]
    [_hiddenWeights addObject:_outputWeights];
    [self _addHiddenDeltaWeights:_outputWeights];
}

/*
 * @ 設定 Output Layer Nets 的 Biases
 */
-(void)addOutputBiases:(NSArray *)_biases
{
    [_outputBiases addObjectsFromArray:_biases];
}

/*
 * @ Random all hidden-net weights, nets biases, output nets biases.
 *   - 亂數設定隱藏層神經元的權重、神經元偏權值、輸出層神經元偏權值
 */
-(void)randomWeights
{
    //先清空歸零
    [_inputWeights removeAllObjects];
    [_hiddenBiases removeAllObjects];
    [_hiddenWeights removeAllObjects];
    //單組輸入向量有多長，就有多少顆輸入層神經元
    NSInteger _inputNetCount  = [[_inputs firstObject] count];
    NSInteger _outputNetCount = [[_outputGoals firstObject] count];
    if( _outputNetCount < 1 )
    {
        _outputNetCount = 1;
    }
    //輸入層到隱藏層的輸入層 Net 數量 = ( (輸入層的 Net 數 * 輸出層 Net 數) ^ 1/2  )
    NSInteger _hiddenNetCount = (int)powf(( _inputNetCount * _outputNetCount ), 0.5f); //_inputNetCount;
    if( _hiddenNetCount < 1 )
    {
        //最少 1 顆
        _hiddenNetCount = 1;
    }
    //亂數給權重值、偏權值
    float _randomMax = 0.5f;
    float _randomMin = -0.5f;
    
    if( [_inputWeights count] < 1 )
    {
        //權重初始化規則 : ( 0.5 / 此層神經元個數 ) ~ ( -0.5 / 此層神經元個數 )，其它層以此類推
        float _inputMax = _randomMax / _inputNetCount;
        float _inputMin = _randomMin / _inputNetCount;
        for( int i=0; i<_inputNetCount; i++ )
        {
            NSMutableArray *_toHiddenWeights = [NSMutableArray new];
            for( int j=0; j<_hiddenNetCount; j++ )
            {
                [_toHiddenWeights addObject:[NSNumber numberWithDouble:[self _randomMax:_inputMax min:_inputMin]]];
            }
            [_inputWeights addObject:_toHiddenWeights];
            [self _addInputDeltaWeights:_toHiddenWeights];
        }
    }
    
    NSInteger _outputCount = [_outputBiases count];
    if( _outputCount < _outputNetCount )
    {
        //輸出層神經元的偏權值
        for( int _i=0; _i<_outputNetCount; _i++ )
        {
            [_outputBiases addObject:[NSNumber numberWithDouble:[self _randomMax:_randomMax min:_randomMin]]];
        }
    }
    
    NSInteger _hiddenCount = [_hiddenWeights count];
    if( _hiddenCount < 1 )
    {
        float _hiddenMax   = _randomMax / _hiddenNetCount;
        float _hiddenMin   = _randomMin / _hiddenNetCount;
        //隱藏層神經元個數 x 輸出層神經元個數，就有幾條至輸出層神經元的權重值
        for( int i=0; i<_hiddenNetCount; i++ )
        {
            NSMutableArray *_toOutputWeights = [NSMutableArray new];
            for( int j=0; j<_outputNetCount; j++ )
            {
                [_toOutputWeights addObject:[NSNumber numberWithDouble:[self _randomMax:_hiddenMax min:_hiddenMin]]];
            }
            [_hiddenWeights addObject:_toOutputWeights];
            [self _addHiddenDeltaWeights:_toOutputWeights];
        }
    }
    
    if( [_hiddenBiases count] < 1 )
    {
        //有幾顆隱藏層神經元，就有幾個偏權值
        for( int i=0; i<_hiddenNetCount; i++ )
        {
            [_hiddenBiases addObject:[NSNumber numberWithDouble:[self _randomMax:_randomMax min:_randomMin]]];
        }
    }
    
}

#pragma --mark Training Public Methods
/*
 * @ Start Training BPN
 *   - Delegate 和 Block 的記憶消耗量在遞迴的實驗下，是一樣的。
 *   - 只單在 dispatch_queue_t 裡跑遞迴，1070 次以後會 Crash，因為 dispatch_queue 的 memory 有限制，改成迭代 1000 次就換下一個 queue 跑 training 就行了。
 */
-(void)training
{
    [self _trainingWithExtraHandler:nil];
}

/*
 * @ Start Training BPN
 *   - And it'll auto save the trained-network when it finished.
 */
-(void)trainingBySave
{
    [self _trainingWithExtraHandler:^
    {
        self._isDoneSave = YES;
    }];
}

/*
 * @ Start Training BPN
 *   - It'll random setup all weights and biases.
 */
-(void)trainingByRandomSettings
{
    [self randomWeights];
    [self training];
}

/*
 * @ Start Training BPN
 *   - It'll random setup all weights and biases, then it'll auto save the trained-network when it finished.
 */
-(void)trainingByRandomWithSave
{
    self._isDoneSave = YES;
    [self randomWeights];
    [self _trainingWithExtraHandler:^
    {
        self._isDoneSave = YES;
    }];
}

/*
 * @ Continue training besides with adding pattern
 */
-(void)trainingWithAddPatterns:(NSArray *)_patterns outputGoals:(NSArray *)_goals
{
    [self addPatterns:_patterns outputGoals:_goals];
    [self continueTraining];
}

/*
 * @ Pause Training BPN
 *   - It'll force stop, and the trained data will keep in network.
 */
-(void)pause
{
    _isTraining = NO;
    _forceStop  = YES;
}

/*
 * @ Continue training
 */
-(void)continueTraining
{
    _forceStop = NO;
    //[self _formatMaxMultiple];
    [self _startTraining];
}

/*
 * @ Reset to initialization
 */
-(void)reset
{
    [self _resetTrainedParameters];
    [self _recoverOriginalParameters];
}

-(void)restart
{
    [self pause];
    [self reset];
    [self training];
}

-(void)directOutputAtInputs:(NSArray *)_rawInputs completion:(void(^)())_completion
{
    if( _rawInputs != nil )
    {
        [_inputs removeAllObjects];
        [_inputs addObject:_rawInputs];
    }
    //取出 Inputs 的第 1 筆 Raw Data
    NSArray *_trainInputs  = [_inputs firstObject];
    dispatch_queue_t queue = dispatch_queue_create("com.krbpn.trained-network", NULL);
    dispatch_async(queue, ^(void){
        [self pause];
        dispatch_async(dispatch_get_main_queue(), ^{
            [self _formatMaxMultiple];
            //將訓練迭代變為 1 次即終止
            //[self recoverTrainedNetwork];
            _limitIteration    = 1;
            _presentIteration = _limitIteration;
            self._hiddenOutputs = [self _sumHiddenLayerOutputsFromInputs:_trainInputs];
            self.outputResults  = [self _sumOutputLayerNetsValue];
            if( self.limitIteration > 0 && self.presentIteration >= self.limitIteration )
            {
                [self _completedTraining];
                if( _completion )
                {
                    _completion();
                }
                return;
            }
        });
    });
}

/*
 * @ 單純使用訓練好的網路作輸出，不跑導傳遞修正網路
 */
-(void)directOutputAtInputs:(NSArray *)_rawInputs
{
    [self directOutputAtInputs:_rawInputs completion:nil];
}

-(void)directOutputAtInputsOnMainThread:(NSArray *)_rawInputs
{
    if( _rawInputs != nil )
    {
        [_inputs removeAllObjects];
        [_inputs addObject:_rawInputs];
    }
    NSArray *_trainInputs  = [_inputs firstObject];
    [self pause];
    [self _formatMaxMultiple];
    _limitIteration    = 1;
    _presentIteration = _limitIteration;
    self._hiddenOutputs = [self _sumHiddenLayerOutputsFromInputs:_trainInputs];
    self.outputResults  = [self _sumOutputLayerNetsValue];
    if( self.limitIteration > 0 && self.presentIteration >= self.limitIteration )
    {
        [self _completedTraining];
        return;
    }
}

#pragma --mark Trained Network Public Methods
/*
 * @ Save the trained-network of BPN to NSUserDefaults
 *   - 儲存訓練後的 BPN Network 至 NSUserDefaults
 *   - 同時會保存原訓練的所有 I/O 數據資料
 */
-(void)saveNetwork
{
    // The key-data sets of trained network.
    KRBPNTrainedNetwork *_bpnNetwork = [[KRBPNTrainedNetwork alloc] init];
    _bpnNetwork.inputs               = _inputs;
    _bpnNetwork.inputWeights         = _inputWeights;
    _bpnNetwork.hiddenWeights        = _hiddenWeights;
    _bpnNetwork.hiddenBiases         = _hiddenBiases;
    _bpnNetwork.outputBiases         = _outputBiases;
    _bpnNetwork.outputResults        = _outputResults;
    _bpnNetwork.outputGoals          = _outputGoals;
    _bpnNetwork.learningRate         = _learningRate;
    _bpnNetwork.convergenceError     = _convergenceError;
    _bpnNetwork.fOfAlpha             = _fOfAlpha;
    _bpnNetwork.limitIteration       = _limitIteration;
    _bpnNetwork.presentIteration     = _presentIteration;
    _bpnNetwork.activeFunction       = _activeFunction;
    _bpnNetwork.learningMode         = _learningMode;
    _bpnNetwork.earlyStopping        = _earlyStopping;
    _bpnNetwork.quickProp            = _quickProp;
    [self removeNetwork];
    _trainedNetwork                  = _bpnNetwork;
    [NSUserDefaults saveTrainedNetwork:_bpnNetwork forKey:_kTrainedNetworkInfo];
}

/*
 * @ Remove the saved trained-netrowk
 */
-(void)removeNetwork
{
    [NSUserDefaults removeValueForKey:_kTrainedNetworkInfo];
    _trainedNetwork = nil;
}

/*
 * @ Recovers trained-network data
 *   - 復原訓練過的 BPN Network 數據資料
 */
-(void)recoverNetwork:(KRBPNTrainedNetwork *)_recoverNetwork
{
    if( _recoverNetwork )
    {
        dispatch_async(dispatch_get_main_queue(), ^
        {
            _inputs             = _recoverNetwork.inputs;
            _inputWeights       = _recoverNetwork.inputWeights;
            _hiddenWeights      = _recoverNetwork.hiddenWeights;
            _hiddenBiases       = _recoverNetwork.hiddenBiases;
            _outputBiases       = _recoverNetwork.outputBiases;
            _outputResults      = _recoverNetwork.outputResults;
            _outputGoals        = _recoverNetwork.outputGoals;
            _learningRate       = _recoverNetwork.learningRate;
            _convergenceError   = _recoverNetwork.convergenceError;
            _fOfAlpha           = _recoverNetwork.fOfAlpha;
            _limitIteration     = _recoverNetwork.limitIteration;
            _presentIteration   = _recoverNetwork.presentIteration;
            _activeFunction     = _recoverNetwork.activeFunction;
            _learningMode       = _recoverNetwork.learningMode;
            _earlyStopping      = _recoverNetwork.earlyStopping;
            _quickProp          = _recoverNetwork.quickProp;
            if( _quickProp != nil &&
                _quickProp.inputLearningMode  == KRQuickPropLearningModeByInputFixed &&
                _quickProp.outputLearningMode == KRQuickPropLearningModeByInputFixed )
            {
                _quickPropFixedRate = _quickProp.inputLearningRate;
            }
            [self removeNetwork];
            _trainedNetwork     = _recoverNetwork;
            [NSUserDefaults saveTrainedNetwork:_trainedNetwork forKey:_kTrainedNetworkInfo];
        });
    }
}

/*
 * @ Recovers saved trained-network of BPN
 *   - 復原已儲存的訓練過的 BPN Network 數據資料
 */
-(void)recoverNetwork
{
    [self recoverNetwork:self.trainedNetwork];
}

#pragma --mark Blocks
-(void)setTrainingCompletion:(KRBPNTrainingCompletion)_theBlock
{
    _trainingCompletion = _theBlock;
}

-(void)setEachIteration:(KRBPNIteration)_theBlock
{
    _eachIteration      = _theBlock;
}

#pragma --mark Getters
-(NSDictionary *)trainedInfo
{
    //代表有不落在 0 ~ 1 之間的值，就回復至原先的數值長度與型態
    if( self._maxMultiple != 1 )
    {
        NSMutableArray *_formatedOutputResults = [NSMutableArray new];
        for( NSNumber *_result in _outputResults )
        {
            //還原每一個 goalValue 當初設定的原同等位同寬度的結果值，即返回原值域
            double _recoveredRetuls = [_result doubleValue] * self._maxMultiple;
            [_formatedOutputResults addObject:[NSNumber numberWithDouble:_recoveredRetuls]];
        }
        self.outputResults     = nil;
        _outputResults         = [[NSArray alloc] initWithArray:_formatedOutputResults];
        [_formatedOutputResults removeAllObjects];
        _formatedOutputResults = nil;
    }
    
    return @{KRBPNTrainedInputWeights  : _inputWeights,
             KRBPNTrainedHiddenWeights : _hiddenWeights,
             KRBPNTrainedHiddenBiases  : _hiddenBiases,
             KRBPNTrainedOutputBiases  : _outputBiases,
             KRBPNTrainedOutputResults : _outputResults,
             KRBPNTrainedIterations    : [NSNumber numberWithInteger:_presentIteration]};
}

-(KRBPNTrainedNetwork *)trainedNetwork
{
    if( !_trainedNetwork )
    {
        _trainedNetwork = [NSUserDefaults trainedNetworkValueForKey:_kTrainedNetworkInfo];
        if ( !_trainedNetwork )
        {
            return nil;
        }
    }
    return _trainedNetwork;
}

-(NSInteger)countHiddenNets
{
    return [_hiddenBiases count];
}

#pragma --mark Setters
-(void)setLearningMode:(KRBPNLearningModes)_mode
{
    _learningMode = _mode;
    if(_learningMode != KRBPNLearningModeByNormal)
    {
        if( _quickProp == nil )
        {
            _quickProp = [KRQuickProp sharedInstance];
        }
        
        switch (_mode)
        {
            case KRBPNLearningModeByQuickPropSmartHybrid:
                [_quickProp setInputFixedRateByRandom];
                _quickProp.outputLearningMode = KRQuickPropLearningModeByOutputDynamic;
                break;
            case KRBPNLearningModeByQuickPropFixed:
                [_quickProp setBothFixedRateByRandom];
                break;
            case KRBPNLearningModeByQuickPropDynamic:
                // ...
                break;
            case KRBPNLearningModeByQuickPropCustomFixed:
                [_quickProp setBothFixedRate:_quickPropFixedRate];
                break;
            default:
                break;
        }
    }
}

-(void)setQuickPropFixedRate:(float)_fixedRate
{
    if(_fixedRate > 0.0f)
    {
        _quickPropFixedRate = _fixedRate;
        if( _quickProp != nil )
        {
            [self setLearningMode:KRBPNLearningModeByQuickPropCustomFixed];
        }
    }
}

@end


