//
//  KRBPN.m
//  BPN
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2014年 Kuo-Ming Lin. All rights reserved.
//

#import "KRBPN.h"
#import "KRBPN+NSUserDefaults.h"

static NSString *_kOriginalInputs           = @"_kOriginalInputs";
static NSString *_kOriginalInputWeights     = @"_kOriginalInputWeights";
static NSString *_kOriginalHiddenWeights    = @"_kOriginalHiddenWeights";
static NSString *_kOriginalHiddenBiases     = @"_kOriginalHiddenBiases";
static NSString *_kOriginalOutputBias       = @"_kOriginalOutputBias";
static NSString *_kOriginalTargetValue      = @"_kOriginalTargetValue";
static NSString *_kOriginalLearningRate     = @"_kOriginalLearningRate";
static NSString *_kOriginalConvergenceError = @"_kOriginalConvergenceError";
static NSString *_kOriginalFOfAlpha         = @"_kOriginalFOfAlpha";
static NSString *_kOriginalLimitGenerations = @"_kOriginalLimitGenerations";

static NSString *_kTrainedNetworkInfo       = @"kTrainedNetworkInfo";

@interface KRBPN ()

//隱藏層的輸出值
@property (nonatomic, strong) NSArray *_hiddenOutputs;
//輸出層的誤差值
@property (nonatomic, strong) NSArray *_outputErrors;
//是否強制中止訓練
@property (nonatomic, assign) BOOL _forceStopTraining;
//原來的設定值
@property (nonatomic, strong) NSMutableDictionary *_originalParameters;
//訓練完就儲存至 NSUserDefaults 裡
@property (nonatomic, assign) BOOL _isDoneSave;

@end

@implementation KRBPN (fixInitials)

-(void)_resetTrainedParameters
{
    self.trainedNetwork      = nil;
    self.trainingGeneration  = 0;
    
    self._hiddenOutputs      = nil;
    self._outputErrors       = nil;
    self._forceStopTraining  = NO;
    self._isDoneSave         = NO;
}

-(void)_initWithVars
{
    self.inputs              = [[NSMutableArray alloc] initWithCapacity:0];
    self.inputWeights        = [[NSMutableArray alloc] initWithCapacity:0];
    self.hiddenWeights       = [[NSMutableArray alloc] initWithCapacity:0];
    self.hiddenBiases        = [[NSMutableArray alloc] initWithCapacity:0];
    self.countHiddens        = 2;
    self.outputBias          = 0.1f;
    self.targetValue         = 1.0f;
    self.learningRate        = 0.8f;
    self.convergenceError    = 0.001f;
    self.fOfAlpha            = 1;
    self.limitGeneration     = 0;
    self.isTraining          = NO;
    self.trainedInfo         = nil;
    
    self.trainingCompletion  = nil;
    self.eachGeneration      = nil;
    
    [self _resetTrainedParameters];
    
    self._originalParameters = [NSMutableDictionary new];
}

@end

@implementation KRBPN (fixMethods)

-(void)_stopTraining
{
    self.isTraining = NO;
    /*
    if( self.trainingCompletion )
    {
        self.trainingCompletion(NO, nil);
    }
     */
}

-(void)_completeTraining
{
    self.isTraining  = NO;
    
    if( self._isDoneSave )
    {
        self._isDoneSave = NO;
        [self saveTrainedNetwork];
    }
    
    if( self.trainingCompletion )
    {
        self.trainingCompletion(YES, self.trainedInfo, self.trainingGeneration);
    }
}

-(void)_copyParametersToTemporary
{
    if( !self._originalParameters )
    {
        self._originalParameters = [NSMutableDictionary new];
    }
    NSMutableDictionary *_originals = self._originalParameters;
    [_originals setObject:[self.inputs copy] forKey:_kOriginalInputs];
    [_originals setObject:[self.inputWeights copy] forKey:_kOriginalInputWeights];
    [_originals setObject:[self.hiddenWeights copy] forKey:_kOriginalHiddenWeights];
    [_originals setObject:[self.hiddenBiases copy] forKey:_kOriginalHiddenBiases];
    [_originals setObject:[NSNumber numberWithDouble:self.outputBias] forKey:_kOriginalOutputBias];
    [_originals setObject:[NSNumber numberWithDouble:self.targetValue] forKey:_kOriginalTargetValue];
    [_originals setObject:[NSNumber numberWithFloat:self.learningRate] forKey:_kOriginalLearningRate];
    [_originals setObject:[NSNumber numberWithDouble:self.convergenceError] forKey:_kOriginalConvergenceError];
    [_originals setObject:[NSNumber numberWithFloat:self.fOfAlpha] forKey:_kOriginalFOfAlpha];
    [_originals setObject:[NSNumber numberWithInteger:self.limitGeneration] forKey:_kOriginalLimitGenerations];
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
            
            self.outputBias       = [[_originals objectForKey:_kOriginalOutputBias] doubleValue];
            self.targetValue      = [[_originals objectForKey:_kOriginalTargetValue] doubleValue];
            self.learningRate     = [[_originals objectForKey:_kOriginalLearningRate] floatValue];
            self.convergenceError = [[_originals objectForKey:_kOriginalConvergenceError] doubleValue];
            self.fOfAlpha         = [[_originals objectForKey:_kOriginalFOfAlpha] floatValue];
            self.limitGeneration  = [[_originals objectForKey:_kOriginalLimitGenerations] integerValue];
        }
    }
}

@end

@implementation KRBPN (fixTrainings)
/*
 * @ 計算隱藏層各個神經元( Nets )的輸出值
 *
 *   - 1. 累加神經元的值 SUM(net)
 *     - net(j) = ( SUM W(ji) * X(i) ) + b(j)
 *
 *   - 2. 代入活化函式   f(net)
 *
 */
-(NSArray *)_sumHiddenLayerNetWeights
{
    /*
     * @ 依公式所說，X(i) 輸入向量應做轉置矩陣運算，但轉置矩陣須耗去多餘效能，
     *   因此，這裡暫不採用直接先轉成轉置矩陣的動作，
     *   而是直接依照資料長度取出同維度的方式來做轉置矩陣。
     *
     * @ 如輸入向量是 X1 = [1, 2, 3]; 的多值型態，就採用線性代數的解法，
     *   將 W(ji) * Xi[0] ~ Xi[n-1] 做相乘後累計的動作，
     *   將公式展開發會變這樣 : 
     *     SUM ( W13 * X1[1, 2, 3] )
     *   將多輸入項變成單一輸入項即可，即 : 
     *     SUM ( ( W13 * X1[0] + W13 * X1[1] + W13 * X1[2] ) ) + ( W23 * X2[0] + W23 * X2[1] + W23 * X2[2] ) ... 以此類推，
     *   將所有的值都加總起來，再進行後續運算。
     *
     */
    //運算完成的 Nets
    NSMutableArray *_nets  = [NSMutableArray new];
    //輸入層要做 SUM 就必須取出同一維度的值做運算
    //先取出輸入向量的陣列長度
    NSInteger _inputLength = [[self.inputs firstObject] count];
    //定義權重的維度
    int _weightDimesion    = -1;
    //依照隱藏層有幾顆神經元(偏權值)，就跑幾次的維度運算
    for( NSNumber *_partialWeight in self.hiddenBiases )
    {
        ++_weightDimesion;
        //net(j)
        NSMutableArray *_fOfNets = [NSMutableArray new];
        //再以同維度做 SUM 方法
        float _sumOfNet = 0;
        for( int i=0; i<_inputLength; i++ )
        {
            
            /*
             * @ 輸入層各向量陣列(值)到隱藏層神經元的權重
             *
             *   inputWeights = @[//W14, W15
             *                    @[@0.2, @-0.3],
             *                    //W24, W25
             *                    @[@0.4, @0.1],
             *                    //W34, W35
             *                    @[@-0.5, @0.2]];
             */
            //取出每一個同維度的輸入層到隱藏層的權重
            int _sameIndex = -1;
            for( NSArray *_everyWeights in self.inputWeights )
            {
                ++_sameIndex;
                /*
                 * @ inputs = @[//X1
                 *              @[@1, @2],
                 *              //X2
                 *              @[@0, @1],
                 *              //X3
                 *              @[@1, @1]];
                 */
                //取出每一個相同維度的輸入向量值，例如 : SUM( x1[0], x2[0], x3[0] ), SUM( x1[99], x2[99], x3[99] ) ...
                NSArray *_sameInputs = [self.inputs objectAtIndex:_sameIndex];
                //取出每陣列 i 維度的值做 SUM 加總
                float _xValue = [[_sameInputs objectAtIndex:i] floatValue];
                //將值與權重相乘後累加，例如 : SUM( w14 x 0.2 + w24 x 0.4 + w34 x -0.5 ), SUM( w15 x -0.3 + w25 x 0.1 + w35 x 0.2 ) ...
                float _weight = [[_everyWeights objectAtIndex:_weightDimesion] floatValue];
                _sumOfNet += _xValue * _weight;
                //NSLog(@"xValue : %f, _weight : %f", _xValue, _weight);
            }
            //NSLog(@"\n\n\n");
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
        //加上同維度的神經元偏權值
        _sumOfNet    += [_partialWeight floatValue];
        //代入活化函式
        float _fOfNet = 1 / ( 1 + powf(M_E, (-(self.fOfAlpha) * _sumOfNet)) );
        //加入計算好的輸入向量值，輸入向量是多少維度，輸出就多少維度，例如 : x1[1, 2, 3]，則 net(j) 就要為 [4, 5, 6] 同等維度。( 這似乎有誤，尚未搞懂 囧 )
        [_fOfNets addObject:[NSNumber numberWithFloat:_fOfNet]];
        [_nets addObject:_fOfNets];
    }
    return ( [_nets count] > 0 ) ? (NSArray *)_nets : nil ;
}

/*
 * @ 計算輸出層神經元的值
 *   - 跟計算隱藏層 [self _sumHiddenLayerNetWeights] 一樣的公式。
 */
-(NSArray *)_sumOutputLayerNetWeights
{
    NSMutableArray *_nets = [NSMutableArray new];
    //net(4) = @[1, 2, 3]; net(5) = @[1, 1, 2]; ...
    self._hiddenOutputs   = [self _sumHiddenLayerNetWeights];
    NSArray *_hiddenNets  = self._hiddenOutputs;
    //NSLog(@"_hiddenNets : %@", _hiddenNets);
    if( _hiddenNets )
    {
        /*
         * @ 隱藏層神經元到輸出層神經元的權重值
         *   
         *   hiddenNetWeights = @[//W46
         *                        @-0.3,
         *                        //W56
         *                        @-0.2];
         */
        NSInteger _inputLength   = [[_hiddenNets firstObject] count];
        NSMutableArray *_fOfNets = [NSMutableArray new];
        for( int i=0; i<_inputLength; i++ )
        {
            float _sumOfNet = 0;
            int _sameIndex  = -1;
            for( NSNumber *_everyWeights in self.hiddenWeights )
            {
                ++_sameIndex;
                NSArray *_sameInputs = [_hiddenNets objectAtIndex:_sameIndex];
                float _xValue = [[_sameInputs objectAtIndex:i] floatValue];
                float _weight = [_everyWeights floatValue];
                _sumOfNet += _xValue * _weight;
                //NSLog(@"xValue : %f, _weight : %f\n\n\n", _xValue, _weight);
            }
            _sumOfNet    += self.outputBias;
            float _fOfNet = 1 / ( 1 + powf(M_E, (-(self.fOfAlpha) * _sumOfNet)) );
            [_fOfNets addObject:[NSNumber numberWithFloat:_fOfNet]];
        }
        [_nets addObject:_fOfNets];
        return _nets;
    }
    return nil;
}

/*
 * @ 計算輸出層神經元( Net )的輸出誤差
 *   - 公式 : Oj x ( 1 - Oj ) x ( Tj - Oj )
 */
-(NSArray *)_calculateOutputError
{
    NSArray *_nets = [self _sumOutputLayerNetWeights];
    //NSLog(@"_nets : %@", _nets);
    if( _nets )
    {
        NSMutableArray *_errors = [NSMutableArray new];
        for( NSArray *_outputs in _nets )
        {
            for( NSNumber *_output in _outputs )
            {
                float _outputValue = [_output floatValue];
                float _errorValue  = _outputValue * ( 1 - _outputValue ) * ( self.targetValue - _outputValue );
                [_errors addObject:[NSNumber numberWithFloat:_errorValue]];
            }
        }
        //NSLog(@"_errors : %@", _errors);
        return _errors;
    }
    return nil;
}

/*
 * @ 計算隱藏層神經元( Nets )的誤差
 *   - 公式 : Oj x ( 1 - Oj ) x Errork x Wjk
 */
-(NSArray *)_calculateNetsError
{
    self._outputErrors   = [self _calculateOutputError];
    NSArray *_outputNets = self._outputErrors;
    if( _outputNets )
    {
        NSMutableArray *_netErrors = [NSMutableArray new];
        //取出輸出層的誤差值倒算回去每一個神經元的誤差值
        for( NSNumber *_outpurError in _outputNets )
        {
            //如果已達收斂誤差，就不再繼續訓練
            float _resultError = [_outpurError floatValue];
            if( _resultError <= self.convergenceError )
            {
                _netErrors = nil;
                break;
                //If break is exception, it will avoid the more exception happens.
                return nil;
            }
            int _netIndex      = -1;
            for( NSArray *_outputs in self._hiddenOutputs )
            {
                ++_netIndex;
                float _netWeight    = [[self.hiddenWeights objectAtIndex:_netIndex] floatValue];
                float _hiddenOutput = [[_outputs firstObject] floatValue];
                //O4    x ( 1 - O4 )    x Error6 x W46
                //0.332 x ( 1 - 0.332 ) x 0.1311 x -0.3
                float _hiddenError  = _hiddenOutput * ( 1 - _hiddenOutput ) * _resultError * _netWeight;
                [_netErrors addObject:[NSNumber numberWithFloat:_hiddenError]];
            }
        }
        return _netErrors;
    }
    return nil;
}

/*
 * @ 更新權重與偏權值
 *   - 公式 : Shita(i) = Shita(i) + learning rate x Error(k)
 *              偏權值 = 偏權值    +   學習速率      x 要修改的誤差值
 */
-(void)_refreshNetsWeights
{
    if( self._forceStopTraining )
    {
        [self _stopTraining];
        return;
    }
    
    self.isTraining = YES;
    ++self.trainingGeneration;
    //隱藏層神經元的輸出誤差值
    NSArray *_hiddenErrors = [self _calculateNetsError];
    //NSLog(@"_hiddenErrors : %@\n\n\n", _hiddenErrors);
    if( _hiddenErrors )
    {
        //先更新輸出層神經元的偏權值
        for( NSNumber *_outputError in self._outputErrors )
        {
            float _errorValue           = [_outputError floatValue];
            self.outputBias = self.outputBias + ( self.learningRate * _errorValue );
            //再更新每一條線的權重
            //先算隱藏層到輸出層的更新權重
            int _hiddenIndex        = -1;
            NSArray *_hiddens = [self.hiddenWeights copy];
            for( NSNumber *_netWeight in _hiddens )
            {
                ++_hiddenIndex;
                NSArray *_outputs   = [self._hiddenOutputs objectAtIndex:_hiddenIndex];
                float _resetWeight  = [_netWeight floatValue] + ( self.learningRate * _errorValue * [[_outputs firstObject] floatValue] );
                //修正隱藏層到輸出層的權重
                [self.hiddenWeights replaceObjectAtIndex:_hiddenIndex withObject:[NSNumber numberWithFloat:_resetWeight]];
            }
            _hiddens = nil;
        }
        
        //接著更新隱藏層神經元的偏權值
        int _netIndex = -1;
        for( NSNumber *_netError in _hiddenErrors )
        {
            ++_netIndex;
            float _netWeight   = [[self.hiddenBiases objectAtIndex:_netIndex] floatValue];
            float _resetWeight = _netWeight + ( self.learningRate * [_netError floatValue] );
            //修正隱藏層偏權值
            [self.hiddenBiases replaceObjectAtIndex:_netIndex withObject:[NSNumber numberWithFloat:_resetWeight]];
        }
        
        //最後更新所有輸入層到隱藏的權重
        NSArray *_weights = [self.inputWeights copy];
        int _inputIndex = -1;
        for( NSArray *_netWeights in _weights )
        {
            ++_inputIndex;
            int _weightIndex = -1;
            NSMutableArray *_resetWeights = [NSMutableArray new];
            for( NSNumber *_everyWeight in _netWeights )
            {
                //每一個權重陣列的元素個數，會等於隱藏層神經元個數
                ++_weightIndex;
                float _netWeight   = [_everyWeight floatValue];
                float _hiddenError = [[_hiddenErrors objectAtIndex:_weightIndex] floatValue];
                //X1 = [1, 2, 3]，則內容總和合 1 + 2 + 3 做 SUM() 方法
                NSArray *_netInputs  = [self.inputs objectAtIndex:_inputIndex];
                float _sumInputValue = 0.0f;
                for( NSNumber *_inputValue in _netInputs )
                {
                    _sumInputValue += [_inputValue floatValue];
                }
                float _resetWeight = _netWeight + ( self.learningRate * _hiddenError * _sumInputValue );
                [_resetWeights addObject:[NSNumber numberWithFloat:_resetWeight]];
                //NSLog(@"_new weight : %f = %f + ( %f * %f * %f )", _resetWeight, _netWeight, self.learningRate, _hiddenError, _sumInputValue);
            }
            //修正 InputWeights 輸入層到隱藏層的權重
            [self.inputWeights replaceObjectAtIndex:_inputIndex withObject:_resetWeights];
        }
        _weights = nil;
        if( self.eachGeneration )
        {
            self.eachGeneration(self.trainingGeneration, self.trainedInfo);
        }
        
    }
    else
    {
        //達到收斂誤差值或出現異常狀況，即停止訓練
        [self _completeTraining];
        return;
    }
    
    //如有指定迭代數 && 當前訓練迭代數 >= 指定迭代數
    if( self.limitGeneration > 0 && self.trainingGeneration >= self.limitGeneration )
    {
        //停止訓練
        [self _completeTraining];
        return;
    }
    
    //未達收斂誤差，則繼續執行訓練
    [self _refreshNetsWeights];
}

@end

@implementation KRBPN

@synthesize inputs              = _inputs;
@synthesize inputWeights        = _inputWeights;
@synthesize hiddenWeights       = _hiddenWeights;
@synthesize hiddenBiases        = _hiddenBiases;
@synthesize countHiddens        = _countHiddens;
@synthesize outputBias          = _outputBias;
@synthesize targetValue         = _targetValue;
@synthesize learningRate        = _learningRate;
@synthesize convergenceError    = _convergenceError;
@synthesize fOfAlpha            = _fOfAlpha;
@synthesize limitGeneration     = _limitGeneration;
@synthesize trainingGeneration  = _trainingGeneration;
@synthesize isTraining          = _isTraining;
@synthesize trainedInfo         = _trainedInfo;
@synthesize trainedNetwork      = _trainedNetwork;

@synthesize trainingCompletion  = _trainingCompletion;
@synthesize eachGeneration      = _eachGeneration;

@synthesize _hiddenOutputs;
@synthesize _outputErrors;
@synthesize _forceStopTraining;
@synthesize _originalParameters;
@synthesize _isDoneSave;

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

#pragma --mark Training Public Methods
/*
 * @ Start Training BPN
 */
-(void)training
{
    dispatch_queue_t queue = dispatch_queue_create("trainingNetwork", NULL);
    dispatch_async(queue, ^(void)
    {
        [self pause];
        [self _resetTrainedParameters];
        [self _copyParametersToTemporary];
        [self _refreshNetsWeights];
    });
}

-(void)trainingDoneSave
{
    self._isDoneSave = YES;
    [self training];
}

/*
 * @ Pause Training BPN
 *   - It'll force stop, and the trained data will keep in network.
 */
-(void)pause
{
    _isTraining        = NO;
    _forceStopTraining = YES;
}

/*
 * @ Continue training
 */
-(void)continueTraining
{
    _forceStopTraining = NO;
    [self _refreshNetsWeights];
}

/*
 * @ Reset to initialization
 */
-(void)reset
{
    [self _resetTrainedParameters];
    [self _recoverOriginalParameters];
}

#pragma --mark Trained Network Public Methods
/*
 * @ Save the trained-network of BPN to NSUserDefaults
 *   - 儲存訓練後的 BPN Network 至 NSUserDefaults
 */
-(void)saveTrainedNetwork
{
    KRBPNTrainedNetwork *_bpnNetwork = [[KRBPNTrainedNetwork alloc] init];
    _bpnNetwork.inputs               = _inputs;
    _bpnNetwork.inputWeights         = _inputWeights;
    _bpnNetwork.hiddenWeights        = _hiddenWeights;
    _bpnNetwork.hiddenBiases         = _hiddenBiases;
    _bpnNetwork.outputBias           = _outputBias;
    _bpnNetwork.targetValue          = _targetValue;
    _bpnNetwork.learningRate         = _learningRate;
    _bpnNetwork.convergenceError     = _convergenceError;
    _bpnNetwork.fOfAlpha             = _fOfAlpha;
    _bpnNetwork.limitGeneration      = _limitGeneration;
    _bpnNetwork.trainingGeneration   = _trainingGeneration;
    [self removeTrainedNetwork];
    _trainedNetwork                  = _bpnNetwork;
    [NSUserDefaults saveTrainedNetwork:_bpnNetwork forKey:_kTrainedNetworkInfo];
}

/*
 * @ Remove the saved trained-netrowk
 */
-(void)removeTrainedNetwork
{
    [NSUserDefaults removeValueForKey:_kTrainedNetworkInfo];
    _trainedNetwork = nil;
}

/*
 * @ Recovers trained-network data
 *   - 復原訓練過的 BPN Network 數據資料
 */
-(void)recoverTrainedNetwork:(KRBPNTrainedNetwork *)_recoverNetwork
{
    if( _recoverNetwork )
    {
        dispatch_async(dispatch_get_main_queue(), ^
        {
            _inputs             = _recoverNetwork.inputs;
            _inputWeights       = _recoverNetwork.inputWeights;
            _hiddenWeights      = _recoverNetwork.hiddenWeights;
            _hiddenBiases       = _recoverNetwork.hiddenBiases;
            _outputBias         = _recoverNetwork.outputBias;
            _targetValue        = _recoverNetwork.targetValue;
            _learningRate       = _recoverNetwork.learningRate;
            _convergenceError   = _recoverNetwork.convergenceError;
            _fOfAlpha           = _recoverNetwork.fOfAlpha;
            _limitGeneration    = _recoverNetwork.limitGeneration;
            _trainingGeneration = _recoverNetwork.trainingGeneration;
            
            [self removeTrainedNetwork];
            _trainedNetwork     = _recoverNetwork;
            [NSUserDefaults saveTrainedNetwork:_trainedNetwork forKey:_kTrainedNetworkInfo];
        });
    }
}

/*
 * @ Recovers saved trained-network of BPN
 *   - 復原已儲存的訓練過的 BPN Network 數據資料
 */
-(void)recoverTrainedNetwork
{
    [self recoverTrainedNetwork:_trainedNetwork];
}

#pragma --mark Blocks
-(void)setTrainingCompletion:(KRBPNTrainingCompletion)_theBlock
{
    _trainingCompletion = _theBlock;
}

-(void)setEachGeneration:(KRBPNEachGeneration)_theBlock
{
    _eachGeneration     = _theBlock;
}

#pragma --mark Getters
-(NSDictionary *)trainedInfo
{
    return @{KRBPNTrainedInfoInputWeights      : _inputWeights,
             KRBPNTrainedInfoHiddenWeights     : _hiddenWeights,
             KRBPNTrainedInfoHiddenBiases      : _hiddenBiases,
             KRBPNTrainedInfoOutputBias        : [NSNumber numberWithDouble:_outputBias],
             KRBPNTrainedInfoTrainedGeneration : [NSNumber numberWithInteger:_trainingGeneration]};
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

@end


