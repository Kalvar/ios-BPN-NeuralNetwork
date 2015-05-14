ios-BPN-NeuralNetwork
=================

Machine Learning (マシンラーニング) in this project, it implemented 3 layers ( Input Layer, Hidden Layer and Output Layer ) neural network (ニューラルネットワーク) and it named Back Propagation Neural Network (BPN). This network can be used in products recommendation (おすすめの商品), user behavior analysis (ユーザーの行動分析), data mining (データマイニング) and data analysis (データ分析).

This project designed for quickly operation for mobile device perform the basic data analysis. If you need help to know how to use this network, just ask me via email.

#### Podfile

```ruby
platform :ios, '7.0'
pod "KRBPN", "~> 1.9"
```

## How to use

``` objective-c
#import "KRBPN.h"

@interface ViewController ()

@property (nonatomic, strong) KRBPN *_krBPN;

@end

@implementation ViewController

@synthesize _krBPN;

//Setups any detail, and 2 outputs, you could set more outputs.
-(void)useSample1
{
    _krBPN.activationFunction = KRBPNActivationFunctionSigmoid;

    //各輸入向量陣列值 & 每一筆輸入向量的期望值( 輸出期望 )，因使用 S 形轉換函數，故 Input 值域須為 [0, 1]，輸出目標為 [0, 1]
    //Add the patterns, the weights connect with hidden layer, the output targets
    [_krBPN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoals:@[@0.7f, @0.8f]];  //Pattern 1
    [_krBPN addPatterns:@[@0, @1, @0.3, @0.9] outputGoals:@[@0.1f, @0.1f]];    //Pattern 2
    [_krBPN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoals:@[@0.95f, @0.9f]]; //Pattern 3
    
    //輸入層各向量值到隱藏層神經元的權重 ( 連結同一個 Net 的就一組一組分開，有幾個 Hidden Net 就會有幾組 )
    //Add pattern-weights in Input layer to Hidden Layer
    [_krBPN addPatternWeights:@[@0.2, @-0.3]]; //W15, W16
    [_krBPN addPatternWeights:@[@0.4, @0.1]];  //W25, W26
    [_krBPN addPatternWeights:@[@-0.5, @0.2]]; //W35, W36
    [_krBPN addPatternWeights:@[@-0.1, @0.3]]; //W45, W46
    
    //隱藏層神經元的偏權值 & 隱藏層神經元到輸出層神經元的權重值
    //Add Hidden Layer biases of nets and the output weights in connect with output layer.
    [_krBPN addHiddenLayerNetBias:-0.4f outputWeights:@[@-0.3f, @0.2f]]; //Net 5 bias, W57, W58 to output layer
    [_krBPN addHiddenLayerNetBias:0.2f outputWeights:@[@-0.2f, @0.5f]];  //Net 6 bias, W67, W68 to output layer
    
    //輸出層神經元偏權值, Net 7, 8
    //Add the output layer biases
    [_krBPN addOutputBiases:@[@0.0f, @0.1f]];
    
    __block typeof(_krBPN) _weakKrBPN = _krBPN;
    //訓練完成時( Training complete )
    [_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKrBPN recoverNetwork];
            [_weakKrBPN directOutputAtInputs:@[@1, @0.1, @0.5, @0.2]];
        }
    }];
    
    [_krBPN training];
    //[_krBPN trainingSave];
}

//Only setups patterns and output goals, and 1 output.
-(void)useSample2
{
    _krBPN.activationFunction = KRBPNActivationFunctionTanh;

    [_krBPN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoals:@[@0.7f]];    //Pattern 1, net 1, 2, 3, 4, and 1 output
    [_krBPN addPatterns:@[@0, @-0.8, @0.3, @-0.9] outputGoals:@[@-0.1f]]; //Pattern 2, same as pattern 1
    [_krBPN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoals:@[@0.9f]];    //Pattern 3, same as pattern 1
    
    __block typeof(_krBPN) _weakKrBPN = _krBPN;
    //訓練完成時( Training complete )
    [_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKrBPN recoverNetwork];
            [_weakKrBPN directOutputAtInputs:@[@0, @-0.8, @0.3, @-0.9]];
        }
    }];
    
    [_krBPN trainingRandom];
    //[_krBPN trainingRandomAndSave];
}

//To learn and verify numbers 0 to 9. And only setups patterns and output goals, and 10 outputs.
-(void)useSample3
{
    _krBPN.activationFunction = KRBPNActivationFunctionSigmoid;

    //1
    [_krBPN addPatterns:@[@0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@1, @0, @0, @0, @0, @0, @0, @0, @0, @0]];
    //2
    [_krBPN addPatterns:@[@1, @0, @0, @0,
                          @1, @1, @1, @1,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @1, @1, @1, @1,
                          @0, @0, @0, @1]
            outputGoals:@[@0, @1, @0, @0, @0, @0, @0, @0, @0, @0]];
    //3
    [_krBPN addPatterns:@[@1, @0, @0, @0,
                          @1, @0, @0, @0,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @1, @0, @0, @0, @0, @0, @0, @0]];
    //4
    [_krBPN addPatterns:@[@1, @1, @1, @1,
                          @1, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @1, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @1, @0,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @1, @0, @0, @0, @0, @0, @0]];
    //5
    [_krBPN addPatterns:@[@1, @1, @1, @1,
                          @1, @0, @0, @0,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @1, @0, @0, @0, @0, @0]];
    //6
    [_krBPN addPatterns:@[@1, @1, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @1, @0, @0, @0, @0]];
    //7
    [_krBPN addPatterns:@[@1, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @1, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @1, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @0, @1, @0, @0, @0]];
    //8
    [_krBPN addPatterns:@[@1, @1, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @1, @0, @0]];
    //9
    [_krBPN addPatterns:@[@1, @1, @1, @1,
                          @1, @0, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @0, @1, @0]];
    //0
    [_krBPN addPatterns:@[@1, @1, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @0, @0,
                          @0, @0, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @0, @0,
                          @0, @0, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @0, @0, @1]];
    
    __block typeof(_krBPN) _weakKrBPN = _krBPN;
    //訓練完成時( Training complete )
    [_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKrBPN recoverNetwork];
            //Verified number " 7 ", and it has some defects.
            [_weakKrBPN directOutputAtInputs:@[@1, @1, @1, @0,
                                               @0, @0, @0, @0,
                                               @0, @1, @0, @0,
                                               @0, @0, @0, @0,
                                               @0, @0, @1, @0,
                                               @0, @0, @0, @0,
                                               @0, @0, @0, @1,
                                               @1, @1, @1, @1,
                                               @1, @1, @1, @1]];
            
        }
    }];
    
    [_krBPN trainingRandom];
    //[_krBPN trainingRandomAndSave];
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    
    _krBPN = [KRBPN sharedNetwork];
    
    //學習速率
    _krBPN.learningRate     = 0.8f;
    //收斂誤差值 ( Normally is 10^-3 or 10^-6 )
    _krBPN.convergenceError = 0.000001f;
    //限制迭代次數
    _krBPN.limitGeneration  = 5000;
    
    //每一次的迭代( Every generation-training )
    [_krBPN setEachGeneration:^(NSInteger times, NSDictionary *trainedInfo){
        NSLog(@"Generation times : %i", times);
        //NSLog(@"Generation result : %f\n\n\n", [trainedInfo objectForKey:KRBPNTrainedInfoOutputResults]);
    }];
    
    //Setup anything by yourself, and 2 outputs.
    [self useSample1];

    //Only setup patterns and output goals, and 1 output.
    [self useSample2];

    //Only setup patterns and output goals, then learning to identify numbers 0 to 9.
    [self useSample3];

}
```

## Version

V1.9.4

## License

MIT.
