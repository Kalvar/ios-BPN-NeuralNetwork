//
//  ViewController.m
//  BPN V2.0.1
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2015年 Kuo-Ming Lin (Kalvar Lin). All rights reserved.
//

#import "ViewController.h"
#import "KRBPN.h"

@interface ViewController ()<KRBPNDelegate>

@property (nonatomic, strong) KRBPN *_krBPN;

@end

@implementation ViewController

@synthesize _krBPN;

//Setups any detail, and 2 outputs, you could set more outputs.
-(void)useSample1
{
    _krBPN.activeFunction = KRBPNActivationBySigmoid;
    
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
}

//Only setups patterns and output goals, and 1 output.
-(void)useSample2
{
    _krBPN.activeFunction = KRBPNActivationByTanh;
    
    [_krBPN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoals:@[@0.7f]];      //Pattern 1
    [_krBPN addPatterns:@[@-0.5, @0.8, @-0.3, @0.9] outputGoals:@[@-0.1f]]; //Pattern 2
    [_krBPN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoals:@[@0.9f]];      //Pattern 3
    
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
            [_weakKrBPN directOutputAtInputs:@[@-0.5, @0.8, @-0.3, @0.9]];
        }
    }];
    
    [_krBPN trainingByRandomSettings];
}

//To learn and verify numbers 0 to 9. And only setups patterns and output goals, and 10 outputs.
-(void)useSample3
{
    _krBPN.activeFunction = KRBPNActivationBySigmoid;
    
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
    
    [_krBPN trainingByRandomSettings];
    //[_krBPN trainingRandomAndSave];
}

//Medical prediction of cancer, 醫療癌症預測
-(void)useSample4
{
    _krBPN.activeFunction = KRBPNActivationBySigmoid;
    
    // ER, PR, Hev-2, K, 治療與否, 每 10 年為 0.1 單位 => 存活率
    [_krBPN addPatterns:@[@0.9, @0.9, @0.0, @0.15, @0, @0.1f] outputGoals:@[@0.12f]];
    [_krBPN addPatterns:@[@0.9, @0.9, @0.0, @0.15, @1, @0.1f] outputGoals:@[@0.12f]];
    
    [_krBPN addPatterns:@[@0.5, @0.4, @0.0, @0.25, @0, @0.1f] outputGoals:@[@0.23f]];
    [_krBPN addPatterns:@[@0.5, @0.4, @0.0, @0.25, @1, @0.1f] outputGoals:@[@0.14f]];
    
    [_krBPN addPatterns:@[@0.0, @0.0, @0.0, @0.3, @0, @0.1f] outputGoals:@[@0.35f]];
    [_krBPN addPatterns:@[@0.0, @0.0, @0.0, @0.3, @1, @0.1f] outputGoals:@[@0.22f]];
    
    [_krBPN addPatterns:@[@0.0, @0.0, @1.0, @0.3, @0, @0.1f] outputGoals:@[@0.3f]];
    [_krBPN addPatterns:@[@0.0, @0.0, @1.0, @0.3, @1, @0.1f] outputGoals:@[@0.12f]];
    
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
            // To predict how many years could be alive with the patients.
            [_weakKrBPN directOutputAtInputs:@[@0.6, @0.6, @0.0, @0.25, @0.0, @0.1f]];
            [_weakKrBPN directOutputAtInputs:@[@0.6, @0.6, @0.0, @0.25, @1.0, @0.1f]];
        }
    }];
    
    [_krBPN trainingByRandomSettings];
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    
	_krBPN = [KRBPN sharedNetwork];
    //_krBPN.delegate = self;
    
    // Learning rate
    _krBPN.learningRate     = 0.8f;
    // Convergence error, 收斂誤差值 ( Normally is 10^-3 or 10^-6 )
    _krBPN.convergenceError = 0.001f;
    // To limit that iterations
    _krBPN.limitIteration   = 1000;
    
    // If you wanna use enhanced theory, and to setup that customized learning rate, you could use this as below :
    //_krBPN.quickPropFixedRate = 0.5f;
    
    // If you wanna directly use QuickProp method to enhance that network training :
    //_krBPN.learningMode = KRBPNLearningModeByQuickPropDynamic;
    
    // If you wanna use Kecman's theory to enhance that network training, please use this :
    //_krBPN.learningMode = KRBPNLearningModeByQuickPropFixed;
    
    // If you wanna use Fahlman's QuickProp combined Kecman's Fixed Rate. Yes, in my experiences this usage method is better than others.
    _krBPN.learningMode = KRBPNLearningModeByQuickPropSmartHybrid;
    
    // 每一次的迭代( Per iteration-training )
    [_krBPN setEachIteration:^(NSInteger times, NSDictionary *trainedInfo){
        NSLog(@"Iterations : %i", times);
        NSLog(@"Result per Iteration : %@", [trainedInfo objectForKey:KRBPNTrainedOutputResults]);
    }];
    
    // Setup anything by yourself, and 2 outputs.
    //[self useSample1];
    
    // Only setup patterns and output goals, and 1 output.
    //[self useSample2];
    
    // Only setup patterns and output goals, then learning to identify numbers 0 to 9.
    [self useSample3];
    
    // 醫療數據預測
    //[self useSample4];
    
    //Start the training, and random the weights, biases, if you use this method that you won't need to setup any weights and biases before.
    //Random means let network to auto setup inputWeights, hiddenBiases, hiddenWeights values.
    //[_krBPN trainingByRandomSettings];
    
    //Random settings as above said, then it will be saved the trained network after done.
    //[_krBPN trainingByRandomWithSave];
    
    //Start the training network, and it won't be saving the trained-network when finished.
    //[_krBPN training];
    
    //Start the training network, and it will auto-saving the trained-network when finished.
    //[_krBPN trainingBySave];
    
    //If you wanna pause the training.
    //[_krBPN pause];
    
    //If you wanna continue the paused training.
    //[_krBPN continueTraining];
    
    //If you wanna reset the network back to initial situation.
    //[_krBPN reset];
    
    //When the training finished, to save the trained-network into NSUserDefaults.
    //[_krBPN saveNetwork];
    
    //If you wanna recover the trained-network data.
    //[_krBPN recoverNetwork];
    
    //To remove the saved trained-network.
    //[_krBPN removeNetwork];
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
}

@end

