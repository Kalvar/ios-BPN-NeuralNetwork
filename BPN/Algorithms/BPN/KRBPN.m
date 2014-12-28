//
//  KRBPN.m
//  BPN V1.2
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2014年 Kuo-Ming Lin (Kalvar). All rights reserved.
//

#import "KRBPN.h"
#import "KRBPN+NSUserDefaults.h"

//To limit the _hiddenNetCount
#define MAX_HIDDEN_NETS_COUNT 20

/*
 * @ 2 種常用的 BPN 模式
 *
 *   - 1. 全部 100 筆都跑完後，總和全部誤差值，再一次回推調整權重與偏權值，這樣才算 1 個迭代。
 *   - 2. 一邊輸入就一邊調整權重與偏權值，一樣要跑完全部 100 筆後，才算 1 個迭代 ( EBP, 誤差導傳遞 )。
 *
 *   2 種都是在每 1 個迭代結束後，判斷 Output Error 是否有達到收斂，如未收斂，就繼續重頭運算，如收斂，就停止訓練 ( 流程參考 P.131 )。
 *
 * @ EBP ( 誤差導傳遞 ) 的流程
 *
 *   - 將 BPN 做一改進，在輸入每一筆的訓練資料時，就「一邊調整權重與偏權值」，直到所有的訓練樣本( 例如 100 筆 )都訓練完後，
 *     才去判斷是否有達到收斂誤差的標準，如有，才停止網路訓練，如果沒有，就重新再代入 100 筆，跑遞迴重新運算一次。
 *
 */
static NSString *_kOriginalInputs           = @"_kOriginalInputs";
static NSString *_kOriginalInputWeights     = @"_kOriginalInputWeights";
static NSString *_kOriginalHiddenWeights    = @"_kOriginalHiddenWeights";
static NSString *_kOriginalHiddenBiases     = @"_kOriginalHiddenBiases";
static NSString *_kOriginalOutputBias       = @"_kOriginalOutputBias";
static NSString *_kOriginalOutputResults    = @"_kOriginalOutputResults";
static NSString *_kOriginalOutputGoals      = @"_kOriginalOutputGoals";
static NSString *_kOriginalLearningRate     = @"_kOriginalLearningRate";
static NSString *_kOriginalConvergenceError = @"_kOriginalConvergenceError";
static NSString *_kOriginalFOfAlpha         = @"_kOriginalFOfAlpha";
static NSString *_kOriginalLimitGenerations = @"_kOriginalLimitGenerations";
//static NSString *_kOriginalMaxMultiple      = @"_kOriginalMaxMultiple";

static NSString *_kTrainedNetworkInfo       = @"kTrainedNetworkInfo";

@interface KRBPN ()

//隱藏層的輸出值
@property (nonatomic, strong) NSArray *_hiddenOutputs;
//當前資料的輸出期望值
@property (nonatomic, assign) double _goalValue;
//輸出層的誤差值
@property (nonatomic, strong) NSArray *_outputErrors;
//是否強制中止訓練
@property (nonatomic, assign) BOOL _forceStopTraining;
//原來的設定值
@property (nonatomic, strong) NSMutableDictionary *_originalParameters;
//訓練完就儲存至 NSUserDefaults 裡
@property (nonatomic, assign) BOOL _isDoneSave;
//記錄當前訓練到哪一組 Input 數據
@property (nonatomic, assign) NSInteger _patternIndex;
//在訓練 goalValue 且其值不在 -1.0f ~ 1.0f 之間時，就使用本值進行相除與回乘原同類型值的動作
@property (nonatomic, assign) NSInteger _maxMultiple;

@end

@implementation KRBPN (fixInitials)

-(void)_resetTrainedParameters
{
    self.outputResults       = nil;
    
    self.trainedNetwork      = nil;
    self.trainingGeneration  = 0;
    
    self._hiddenOutputs      = nil;
    self._outputErrors       = nil;
    self._forceStopTraining  = NO;
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
    self.outputBias          = 0.1f;
    self.outputGoals         = nil;
    self.learningRate        = 0.8f;
    self.convergenceError    = 0.001f;
    self.fOfAlpha            = 1;
    self.limitGeneration     = 0;
    self.isTraining          = NO;
    self.trainedInfo         = nil;
    
    self.trainingCompletion  = nil;
    self.eachGeneration      = nil;
    
    [self _resetTrainedParameters];
    
    self._maxMultiple        = 1;
    self._goalValue          = 1.0f;
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

-(void)_completedTraining
{
    self.isTraining  = NO;
    if( self._isDoneSave )
    {
        self._isDoneSave = NO;
        [self saveTrainedNetwork];
    }
    
    if( self.delegate )
    {
        if( [self.delegate respondsToSelector:@selector(krBPNDidTrainFinished:trainedInfo:totalTimes:)] )
        {
            [self.delegate krBPNDidTrainFinished:self trainedInfo:self.trainedInfo totalTimes:self.trainingGeneration];
        }
    }
    
    if( self.trainingCompletion )
    {
        self.trainingCompletion(YES, self.trainedInfo, self.trainingGeneration);
    }
}

-(void)_printEachGeneration
{
    if( self.delegate )
    {
        if( [self.delegate respondsToSelector:@selector(krBPNEachGeneration:trainedInfo:times:)] )
        {
            [self.delegate krBPNEachGeneration:self trainedInfo:self.trainedInfo times:self.trainingGeneration];
        }
    }
    
    if( self.eachGeneration )
    {
        self.eachGeneration(self.trainingGeneration, self.trainedInfo);
    }
}

-(void)_copyParametersToTemporary
{
    if( !self._originalParameters )
    {
        self._originalParameters = [NSMutableDictionary new];
    }
    else
    {
        [self._originalParameters removeAllObjects];
    }
    NSMutableDictionary *_originals = self._originalParameters;
    [_originals setObject:[self.inputs copy] forKey:_kOriginalInputs];
    [_originals setObject:[self.inputWeights copy] forKey:_kOriginalInputWeights];
    [_originals setObject:[self.hiddenWeights copy] forKey:_kOriginalHiddenWeights];
    [_originals setObject:[self.hiddenBiases copy] forKey:_kOriginalHiddenBiases];
    [_originals setObject:[NSNumber numberWithDouble:self.outputBias] forKey:_kOriginalOutputBias];
    [_originals setObject:[self.outputGoals copy] forKey:_kOriginalOutputGoals];
    [_originals setObject:[NSNumber numberWithFloat:self.learningRate] forKey:_kOriginalLearningRate];
    [_originals setObject:[NSNumber numberWithDouble:self.convergenceError] forKey:_kOriginalConvergenceError];
    [_originals setObject:[NSNumber numberWithFloat:self.fOfAlpha] forKey:_kOriginalFOfAlpha];
    [_originals setObject:[NSNumber numberWithInteger:self.limitGeneration] forKey:_kOriginalLimitGenerations];
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
            
            self.outputBias       = [[_originals objectForKey:_kOriginalOutputBias] doubleValue];
            self.outputGoals      = [NSArray arrayWithArray:[_originals objectForKey:_kOriginalOutputGoals]];
            self.learningRate     = [[_originals objectForKey:_kOriginalLearningRate] floatValue];
            self.convergenceError = [[_originals objectForKey:_kOriginalConvergenceError] doubleValue];
            self.fOfAlpha         = [[_originals objectForKey:_kOriginalFOfAlpha] floatValue];
            self.limitGeneration  = [[_originals objectForKey:_kOriginalLimitGenerations] integerValue];
            
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
     */
    //srand((int)time(NULL));
    //double _random = ((double)rand() / RAND_MAX) * (_maxValue - _minValue) + _minValue;
    //RAND_MAX 是 0x7fffffff (2147483647)，而 arc4random() 返回的最大值则是 0x100000000 (4294967296)，故 * 2.0f 待除，或使用自訂義 ARC4RANDOM_MAX      0x100000000
    double _random = ((double)arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
    //NSLog(@"_random : %lf", _random);
    return _random;
}

/*
 * @ 先針對 Output Goals 輸出期望值集合做資料正規化
 *   - 以免因為少設定映射的期望結果而 Crash
 */
-(void)_formatOutputGoals
{
    NSInteger _goalCount  = [self.outputGoals count];
    NSInteger _inputCount = [self.inputs count];
    //輸出期望值組數 < 輸入向量的 Pattern 組數
    if( _goalCount < _inputCount )
    {
        //將缺少的部份用 0.0f 補滿
        NSMutableArray *_goals = [[NSMutableArray alloc] initWithArray:self.outputGoals];
        NSInteger _diffCount   = _inputCount - _goalCount;
        for( int i=0; i<_diffCount; i++ )
        {
            [_goals addObject:@0.0f];
        }
        self.outputGoals = [[NSArray alloc] initWithArray:_goals];
        [_goals removeAllObjects];
        _goals = nil;
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
 *   - 2. 代入活化函式 f(net), LMS 最小均方法
 *
 */
-(NSArray *)_sumHiddenLayerNetWeightsFromInputs:(NSArray *)_inputs
{
    //運算完成的 Nets
    NSMutableArray *_nets  = [NSMutableArray new];
    //輸入層要做 SUM 就必須取出同一維度的值做運算
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
        /*
         * @ inputs = @[//X1
         *              @[@1, @2, @-1],
         *              //X2
         *              @[@0, @1, @1],
         *              //X3
         *              @[@1, @1, @-2]];
         */
        //有幾個 Input 就有幾個 Weight
        //取出每一個輸入值( Ex : X1 轉置矩陣後的輸入向量 [1, 2, -1] )
        int _inputIndex = -1;
        for( NSNumber *_xi in _inputs )
        {
            ++_inputIndex;
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
            NSArray *_everyWeights = [self.inputWeights objectAtIndex:_inputIndex];
            //將值與權重相乘後累加，例如 : SUM( w14 x 0.2 + w24 x 0.4 + w34 x -0.5 ), SUM( w15 x -0.3 + w25 x 0.1 + w35 x 0.2 ) ...
            float _weight  = [[_everyWeights objectAtIndex:_weightDimesion] floatValue];
            _sumOfNet     += [_xi floatValue] * _weight;
            //NSLog(@"xValue : %f, _weight : %f", [_xi floatValue], _weight);
        }
        //NSLog(@"\n\n\n");
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
        //代入活化函式 ( 用 1 除之，則預設限定範圍在 ~ 1.0 以下 )
        float _fOfNet = 1 / ( 1 + powf(M_E, (-(self.fOfAlpha) * _sumOfNet)) );
        //加入計算好的輸入向量值，輸入向量是多少維度，輸出就多少維度，例如 : x1[1, 2, 3]，則 net(j) 就要為 [4, 5, 6] 同等維度。( 這似乎有誤，尚未搞懂 囧 )
        [_fOfNets addObject:[NSNumber numberWithFloat:_fOfNet]];
        [_nets addObject:_fOfNets];
    }
    return ( [_nets count] > 0 ) ? (NSArray *)_nets : nil ;
}

/*
 * @ 計算輸出層神經元的值
 *   - 跟計算隱藏層 [self _sumHiddenLayerNetWeightsFromInputs:] 一樣的公式。
 */
-(NSArray *)_sumOutputLayerNetsValue
{
    NSMutableArray *_nets = [NSMutableArray new];
    NSArray *_hiddenNets  = self._hiddenOutputs;
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
            //float _fOfNet = 3 / ( 1 + powf(M_E, (-(self.fOfAlpha) * _sumOfNet)) );
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
    self.outputResults = [self _sumOutputLayerNetsValue];
    NSArray *_nets     = self.outputResults;
    //NSLog(@"output net values : %@", _nets);
    if( _nets )
    {
        NSMutableArray *_errors = [NSMutableArray new];
        for( NSArray *_outputs in _nets )
        {
            for( NSNumber *_output in _outputs )
            {
                //取出輸出層神經元的輸出結果
                float _outputValue = [_output floatValue];
                //計算與期望值的誤差
                float _errorValue  = _outputValue * ( 1 - _outputValue ) * ( self._goalValue - _outputValue );
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
            float _resultError = [_outpurError floatValue];
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
-(BOOL)_refreshNetsWeights
{
    if( self._forceStopTraining )
    {
        [self _stopTraining];
        return NO;
    }
    
    self.isTraining = YES;
    BOOL _onePatternTrained = NO;
    //隱藏層神經元的輸出誤差值
    NSArray *_hiddenErrors = [self _calculateNetsError];
    //NSLog(@"_hiddenErrors : %@\n\n\n", _hiddenErrors);
    if( _hiddenErrors )
    {
        //先更新輸出層神經元的偏權值
        for( NSNumber *_outputError in self._outputErrors )
        {
            float _errorValue = [_outputError floatValue];
            self.outputBias   = self.outputBias + ( self.learningRate * _errorValue );
            //再更新每一條線的權重
            //先算隱藏層到輸出層的更新權重
            int _hiddenIndex  = -1;
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
                float _inputValue  = [[[self.inputs objectAtIndex:self._patternIndex] objectAtIndex:_inputIndex] floatValue];
                float _resetWeight = _netWeight + ( self.learningRate * _hiddenError * _inputValue );
                [_resetWeights addObject:[NSNumber numberWithFloat:_resetWeight]];
                //NSLog(@"_new weight : %f = %f + ( %f * %f * %f )", _resetWeight, _netWeight, self.learningRate, _hiddenError, _inputValue);
            }
            //修正 InputWeights 輸入層到隱藏層的權重
            [self.inputWeights replaceObjectAtIndex:_inputIndex withObject:_resetWeights];
        }
        _weights           = nil;
        _onePatternTrained = YES;
    }
    
    return _onePatternTrained;
}

//公式化訓練用的期望值
-(void)_formatMaxMultiple
{
    //先找出期望值的最大絕對值
    NSNumber *_max  = [self.outputGoals valueForKeyPath:@"@max.self"];
    NSNumber *_min  = [self.outputGoals valueForKeyPath:@"@min.self"];
    double _fabsMax = fabs(_max.doubleValue);
    double _fabsMin = fabs(_min.doubleValue);
    double _realMax = MAX(_fabsMax, _fabsMin);
    if( _realMax > 1.0f )
    {
        self._maxMultiple  = 10 * ( (int)log10(_realMax) + 1 );
    }
}

-(void)_startTraining
{
    ++self.trainingGeneration;
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
    //先正規化處理資料，以避免異常狀況發生
    [self _formatOutputGoals];
    //開始代入 X1, X2 ... Xn 各組的訓練資料
    for( NSArray *_inputs in self.inputs )
    {
        ++self._patternIndex;
        /*
         * @ 每一筆輸入向量( 每組訓練的 Pattern )都會有自己的輸出期望值
         *   － 例如 : 
         *           輸入 X1[1, 0, 0]，其期望輸出為 1.0
         *           輸入 X2[0, 1, 0]，其期望輸出為 2.0
         *           輸入 X3[0, 0, 1]，其期望輸出為 3.0
         *      以此類推。
         */
        //不論正負號都先轉成絕對值，我只要求得除幾位數變成小數點
        //NSLog(@"%i", (int)log10f(-81355.555)); //-2147483648
        //NSLog(@"%i", (int)log10f(81355.555)); //4 個 10 倍
        self._goalValue     = [[self.outputGoals objectAtIndex:self._patternIndex] doubleValue] / self._maxMultiple;
        self._hiddenOutputs = [self _sumHiddenLayerNetWeightsFromInputs:_inputs];
        //NSLog(@"\n\n_goalValue : %lf, _hiddenOutputs : %@\n\n\n", self._goalValue, self._hiddenOutputs);
        //更新權重失敗，代表訓練異常，中止 !
        if ( ![self _refreshNetsWeights] )
        {
            //考慮，是否要記錄訓練到哪一筆，等等再繼續 ?
            //要繼續的話應該要重頭再來才是 ?
            break;
        }
    }
    
    //如有指定迭代數 && 當前訓練迭代數 >= 指定迭代數
    if( self.limitGeneration > 0 && self.trainingGeneration >= self.limitGeneration )
    {
        //停止訓練
        [self _completedTraining];
        return;
    }
    
    //檢查是否收斂
    BOOL _isGoalError = NO;
    for( NSNumber *_outpurError in self._outputErrors )
    {
        //使用絕對值來做誤差比較
        float _resultError = fabsf( [_outpurError floatValue] );
        //如果已達收斂誤差，就不再繼續訓練
        if( _resultError <= self.convergenceError )
        {
            _isGoalError = YES;
            break;
        }
    }
    
    if( _isGoalError )
    {
        //達到收斂誤差值或出現異常狀況，即停止訓練
        [self _completedTraining];
        return;
    }
    else
    {
        //全部數據都訓練完了，才為 1 迭代
        [self _printEachGeneration];
        //未達收斂誤差，則繼續執行訓練
        [self _startTraining];
    }
}

-(void)_trainingWithExtraSetupHandler:(void(^)())_extraSetupHandler
{
    //DISPATCH_QUEUE_CONCURRENT
    dispatch_queue_t queue = dispatch_queue_create("com.krbpn.train-network", NULL);
    dispatch_async(queue, ^(void)
    {
        [self pause];
        [self _resetTrainedParameters];
        if( _extraSetupHandler )
        {
            _extraSetupHandler();
        }
        [self _copyParametersToTemporary];
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
@synthesize outputBias          = _outputBias;
@synthesize outputResults       = _outputResults;
@synthesize outputGoals         = _outputGoals;
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
@synthesize _goalValue;
@synthesize _outputErrors;
@synthesize _forceStopTraining;
@synthesize _originalParameters;
@synthesize _isDoneSave;
@synthesize _patternIndex;
@synthesize _maxMultiple;

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
 * @ Random all hidden-net weights, net biases, output net bias.
 *   - 亂數設定隱藏層神經元的權重、神經元偏權值、輸出層神經元偏權值
 *
 * @ 如果不指定神經元的權重值，那就自行依照「輸入向量的神經元個數做平方」，也就是輸入層如 2 個神經元，
 *   則隱藏層神經元就預設為 2^2 = 4 個 ( 參考 ANFIS ) 的作法，而每一個輸入層到隱藏層的權重值，
 *   就直接亂數給 -1.0 ~ 1.0 之間的值，而每一個神經元的偏權值也是亂數給 -1.0 ~ 1.0。
 */
-(void)randomWeights
{
    //先清空歸零
    [_inputWeights removeAllObjects];
    [_hiddenBiases removeAllObjects];
    [_hiddenWeights removeAllObjects];
    //亂數給權重值、偏權值
    CGFloat _randomMax       = 1.0f;
    CGFloat _randomMin       = -1.0f;
    //單組輸入向量有多長，就有多少顆輸入層神經元
    NSInteger _inputNetCount = [[_inputs firstObject] count];
    //神經元顆數乘平方即為輸入層到隱藏層的輸入權重總數
    //後續也能考慮當神經元數目過多時，直接除以 2 以減低訓練時間和負擔
    NSInteger _hiddenNetCount = _inputNetCount;
    if( _hiddenNetCount < 1 )
    {
        //最少 1 顆
        _hiddenNetCount = 1;
    }
    
    //隱藏層神經元 MAX is 20
    if( _hiddenNetCount > MAX_HIDDEN_NETS_COUNT )
    {
        _hiddenNetCount = MAX_HIDDEN_NETS_COUNT;
    }
    
    if( [_inputWeights count] < 1 )
    {
        //計算共有幾條隱藏層的權重線
        //NSInteger _totalLines = _hiddenNetCount * _inputNetCount;
        for( int i=0; i<_inputNetCount; i++ )
        {
            NSMutableArray *_netWeights = [NSMutableArray new];
            for( int j=0; j<_hiddenNetCount; j++ )
            {
                [_netWeights addObject:[NSNumber numberWithDouble:[self _randomMax:_randomMax min:_randomMin]]];
            }
            [_inputWeights addObject:_netWeights];
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
    
    if( [_hiddenWeights count] < 1 )
    {
        //有幾顆隱藏層神經元，就有幾條至輸出層神經元的權重值
        for( int i=0; i<_hiddenNetCount; i++ )
        {
            [_hiddenWeights addObject:[NSNumber numberWithDouble:[self _randomMax:_randomMax min:_randomMin]]];
        }
    }
    
    //輸出層神經元的偏權值
    _outputBias = [self _randomMax:_randomMax min:_randomMin];
    
}



/*
 * @ Start Training BPN
 *   - Delegate 和 Block 的記憶消耗量在遞迴的實驗下，是一樣的。
 *   - 只單在 dispatch_queue_t 裡跑遞迴，1070 次以後會 Crash，因為 dispatch_queue 的 memory 有限制，改成迭代 1000 次就換下一個 queue 跑 training 就行了。
 */
-(void)training
{
    [self _trainingWithExtraSetupHandler:nil];
}

/*
 * @ Start Training BPN
 *   - And it'll auto save the trained-network when it finished.
 */
-(void)trainingDoneSave
{
    [self _trainingWithExtraSetupHandler:^
    {
        self._isDoneSave = YES;
    }];
}

/*
 * @ Start Training BPN
 *   - It'll random setup all weights and biases.
 */
-(void)trainingWithRandom
{
    [self randomWeights];
    [self training];
}

/*
 * @ Start Training BPN
 *   - It'll random setup all weights and biases, then it'll auto save the trained-network when it finished.
 */
-(void)trainingWithRandomAndSave
{
    self._isDoneSave = YES;
    [self randomWeights];
    [self _trainingWithExtraSetupHandler:^
    {
        self._isDoneSave = YES;
    }];
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

/*
 * @ 
 *   - 單純使用訓練好的網路作輸出，不跑導傳遞的訓練方法
 */
-(void)useTrainedNetworkToOutput
{
    NSArray *_trainInputs  = [_inputs firstObject];
    dispatch_queue_t queue = dispatch_queue_create("com.krbpn.trained-network", NULL);
    dispatch_async(queue, ^(void){
        [self pause];
        dispatch_async(dispatch_get_main_queue(), ^{
            [self _formatMaxMultiple];
            //將訓練迭代變為 1 次即終止
            //[self recoverTrainedNetwork];
            _limitGeneration    = 1;
            _trainingGeneration = _limitGeneration;
            self._hiddenOutputs = [self _sumHiddenLayerNetWeightsFromInputs:_trainInputs];
            self.outputResults  = [self _sumOutputLayerNetsValue];
            if( self.limitGeneration > 0 && self.trainingGeneration >= self.limitGeneration )
            {
                [self _completedTraining];
                return;
            }
        });
    });
    
}

#pragma --mark Trained Network Public Methods
/*
 * @ Save the trained-network of BPN to NSUserDefaults
 *   - 儲存訓練後的 BPN Network 至 NSUserDefaults
 *   - 同時會保存原訓練的所有 I/O 數據資料
 */
-(void)saveTrainedNetwork
{
    KRBPNTrainedNetwork *_bpnNetwork = [[KRBPNTrainedNetwork alloc] init];
    _bpnNetwork.inputs               = _inputs;
    _bpnNetwork.inputWeights         = _inputWeights;
    _bpnNetwork.hiddenWeights        = _hiddenWeights;
    _bpnNetwork.hiddenBiases         = _hiddenBiases;
    _bpnNetwork.outputBias           = _outputBias;
    _bpnNetwork.outputResults        = _outputResults;
    _bpnNetwork.outputGoals          = _outputGoals;
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
            _outputResults      = _recoverNetwork.outputResults;
            _outputGoals        = _recoverNetwork.outputGoals;
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
    //[self recoverTrainedNetwork:_trainedNetwork];
    [self recoverTrainedNetwork:self.trainedNetwork];
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
    //代表有不落在 0 ~ 1 之間的值，就回復至原先的數值長度與型態
    if( self._maxMultiple != 1 )
    {
        NSMutableArray *_formatedOutputResults = [NSMutableArray new];
        for( NSArray *_groupResults in _outputResults )
        {
            for( NSNumber *_result in _groupResults )
            {
                //還原每一個 goalValue 當初設定的原同等位同寬度的結果值
                double _recoveredRetuls = [_result doubleValue] * self._maxMultiple;
                [_formatedOutputResults addObject:[NSNumber numberWithDouble:_recoveredRetuls]];
            }
        }
        self.outputResults     = nil;
        _outputResults         = [[NSArray alloc] initWithArray:_formatedOutputResults];
        [_formatedOutputResults removeAllObjects];
        _formatedOutputResults = nil;
    }
    
    return @{KRBPNTrainedInfoInputWeights      : _inputWeights,
             KRBPNTrainedInfoHiddenWeights     : _hiddenWeights,
             KRBPNTrainedInfoHiddenBiases      : _hiddenBiases,
             KRBPNTrainedInfoOutputBias        : [NSNumber numberWithDouble:_outputBias],
             KRBPNTrainedInfoOutputResults     : _outputResults,
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

-(NSInteger)countHiddenNets
{
    return [_hiddenBiases count];
}

@end


