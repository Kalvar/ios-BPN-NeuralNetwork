ios-BPN-NeuralNetwork
=================

Machine Learning (マシンラーニング) in this project, it implemented 3 layers ( Input Layer, Hidden Layer and Output Layer ) neural network (ニューラルネットワーク) and implemented Back Propagation Neural Network (BPN), QuickProp theory and Kecman's theory (EDBD). KRBPN can be used in products recommendation (おすすめの商品), user behavior analysis (ユーザーの行動分析), data mining (データマイニング) and data analysis (データ分析).

This project designed for quickly operation for mobile device perform the basic data analysis. If you need help to know how to use this network, just ask me via email.

#### Podfile

```ruby
platform :ios, '7.0'
pod "KRBPN", "~> 2.0.6"
```

## How to use

#### Import

``` objective-c
#import "KRBPN.h"
```

#### Common Settings

``` objective-c
// Use singleton or [[KRBPN alloc] init]
_krBPN = [KRBPN sharedNetwork];

// Learning rate
_krBPN.learningRate     = 0.8f;

// Convergence error, 收斂誤差值 ( Normally is 10^-3 or 10^-6 )
_krBPN.convergenceError = 0.001f;

// Limit iterations
_krBPN.limitIteration   = 1000;

// If you wanna up to your wishing to decide that initital hidden-net counts, you could use this : 
_krBPN.hiddenNets       = 5;

// If you wanna use enhanced theory, and to setup that customized learning rate, you could use this as below :
_krBPN.quickPropFixedRate = 0.5f;

// If you wanna directly use QuickProp method to enhance that network training :
_krBPN.learningMode = KRBPNLearningModeByQuickPropDynamic;

// If you wanna use Kecman's theory to enhance that network training, please use this :
_krBPN.learningMode = KRBPNLearningModeByQuickPropFixed;

// If you wanna use Fahlman's QuickProp combined Kecman's Fixed Rate. Yes, in my experiences this usage method is better than others.
_krBPN.learningMode = KRBPNLearningModeByQuickPropSmartHybrid;

// Per iteration-training block
[_krBPN setEachIteration:^(NSInteger times, NSDictionary *trainedInfo){
    NSLog(@"Iterations : %i", times);
    NSLog(@"Result per Iteration : %@", [trainedInfo objectForKey:KRBPNTrainedOutputResults]);
}];
```

#### Sample 1

To use Sigmoid active function and customize all weights & biases.

``` objective-c
// Sigmoid is that values range in [0.0, 1.0] and the outputs too.
_krBPN.activeFunction = KRBPNActivationBySigmoid;

// Add the 3 patterns, the weights connect with hidden layer, the output targets
[_krBPN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoals:@[@0.7f, @0.8f]];
[_krBPN addPatterns:@[@0, @1, @0.3, @0.9] outputGoals:@[@0.1f, @0.1f]];  
[_krBPN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoals:@[@0.95f, @0.9f]];

// Add input to hidden layer that pattern-weights
// Here is 2 hidden nets
[_krBPN addPatternWeights:@[@0.2, @-0.3]];
[_krBPN addPatternWeights:@[@0.4, @0.1]];  
[_krBPN addPatternWeights:@[@-0.5, @0.2]]; 
[_krBPN addPatternWeights:@[@-0.1, @0.3]]; 

// Add hidden layer biases of nets and the output weights
[_krBPN addHiddenLayerNetBias:-0.4f outputWeights:@[@-0.3f, @0.2f]]; 
[_krBPN addHiddenLayerNetBias:0.2f outputWeights:@[@-0.2f, @0.5f]]; 

// Add 2 output nets biases
[_krBPN addOutputBiases:@[@0.0f, @0.1f]];

__block typeof(_krBPN) _weakKrBPN = _krBPN;
[_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
    if( success )
    {
        NSLog(@"Training done with total times : %i", totalTimes);
        NSLog(@"TrainedInfo : %@", trainedInfo);
        
        //Start in checking the network is correctly trained.
        NSLog(@"======== Start in Verification ========");

        [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"Verified TrainedInfo : %@", trainedInfo);
        }];
        
        [_weakKrBPN recoverNetwork];
        [_weakKrBPN directOutputAtInputs:@[@1, @0.1, @0.5, @0.2]];
    }
}];

[_krBPN training];
```

#### Sample 2

This sample used tanh() that range in [-1.0, 1.0] an active function, and we directly random all settings (weights, biases, net counts).

``` objective-c
_krBPN.activeFunction = KRBPNActivationByTanh;

[_krBPN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoals:@[@0.7f]];    
[_krBPN addPatterns:@[@-0.5, @0.8, @-0.3, @0.9] outputGoals:@[@-0.1f]]; 
[_krBPN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoals:@[@0.9f]];     

__block typeof(_krBPN) _weakKrBPN = _krBPN;
[_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
    if( success )
    {
        NSLog(@"Training done with total times : %i", totalTimes);
        NSLog(@"TrainedInfo : %@", trainedInfo);
        
        //Start in checking the network is correctly trained.
        NSLog(@"======== Start in Verification ========");

        [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"Verified TrainedInfo : %@", trainedInfo);
        }];
        
        [_weakKrBPN recoverNetwork];
        [_weakKrBPN directOutputAtInputs:@[@-0.5, @0.8, @-0.3, @0.9]];
    }
}];

[_krBPN trainingByRandomSettings];
```

#### Sample 3

Identify numbers 0 to 9, this training has 10 outputs (same as 10 classified groups). Number 1 is [1, 0, 0, 0, ..., 0], number 3 is [0, 0, 1, 0, ...., 0].

``` objective-c
_krBPN.activeFunction = KRBPNActivationBySigmoid;
    
// Number 1
[_krBPN addPatterns:@[@0, @0, @0, @0, @0, @0, @0, @0, @0, @0, @0, @0, 
                      @0, @0, @0, @0, @0, @0, @0, @0, @0, @0, @0, @0,
                      @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1]
        outputGoals:@[@1, @0, @0, @0, @0, @0, @0, @0, @0, @0]];

// Number 2
[_krBPN addPatterns:@[@1, @0, @0, @0, @1, @1, @1, @1, @1, @1, @0, @0,
                      @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                      @0, @0, @1, @1, @1, @1, @1, @1, @0, @0, @0, @1]
        outputGoals:@[@0, @1, @0, @0, @0, @0, @0, @0, @0, @0]];

// Number 3
[_krBPN addPatterns:@[@1, @0, @0, @0, @1, @0, @0, @0, @1, @1, @0, @0,
                      @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                      @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1, @1]
        outputGoals:@[@0, @0, @1, @0, @0, @0, @0, @0, @0, @0]];

// Number 4
[_krBPN addPatterns:@[@1, @1, @1, @1, @1, @0, @0, @0, @0, @0, @0, @0,
                      @0, @1, @0, @0, @0, @0, @0, @0, @0, @0, @1, @0,
                      @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1]
        outputGoals:@[@0, @0, @0, @1, @0, @0, @0, @0, @0, @0]];

// Number 5
[_krBPN addPatterns:@[@1, @1, @1, @1, @1, @0, @0, @0, @1, @1, @0, @0,
                      @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                      @0, @0, @1, @1, @0, @0, @0, @1, @1, @1, @1, @1]
        outputGoals:@[@0, @0, @0, @0, @1, @0, @0, @0, @0, @0]];

// Number 6
[_krBPN addPatterns:@[@1, @1, @1, @1, @1, @1, @1, @1, @1, @1, @0, @0,
                      @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                      @0, @0, @1, @1, @0, @0, @0, @1, @1, @1, @1, @1]
        outputGoals:@[@0, @0, @0, @0, @0, @1, @0, @0, @0, @0]];

// Number 7
[_krBPN addPatterns:@[@1, @0, @0, @0, @0, @0, @0, @0, @0, @1, @0, @0,
                      @0, @0, @0, @0, @0, @0, @1, @0, @0, @0, @0, @0,
                      @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1]
        outputGoals:@[@0, @0, @0, @0, @0, @0, @1, @0, @0, @0]];

// Number 8
[_krBPN addPatterns:@[@1, @1, @1, @1, @1, @1, @1, @1, @1, @1, @0, @0,
                      @0, @1, @0, @0, @0, @1, @1, @0, @0, @0, @1, @0,
                      @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1, @1]
        outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @1, @0, @0]];

// Number 9
[_krBPN addPatterns:@[@1, @1, @1, @1, @1, @0, @0, @0, @0, @1, @0, @0,
                      @0, @1, @0, @0, @0, @0, @1, @0, @0, @0, @1, @0,
                      @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1]
        outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @0, @1, @0]];

// Number 0
[_krBPN addPatterns:@[@1, @1, @1, @1, @1, @1, @1, @1, @1, @1, @0, @0,
                      @0, @0, @0, @0, @0, @1, @1, @0, @0, @0, @0, @0,
                      @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1, @1]
        outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @0, @0, @1]];

__block typeof(_krBPN) _weakKrBPN = _krBPN;
[_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
    if( success )
    {
        NSLog(@"Training done with total times : %i", totalTimes);
        NSLog(@"TrainedInfo : %@", trainedInfo);
        
        //Start in checking the network is correctly trained.
        NSLog(@"======== Start in Verification ========");

        [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"Verified TrainedInfo : %@", trainedInfo);
        }];
        
        [_weakKrBPN recoverNetwork];

        //Verified number " 7 ", and it has some defects.
        [_weakKrBPN directOutputAtInputs:@[@1, @1, @1, @0, @0, @0, @0, @0, @0, @1, @0, @0,
                                           @0, @0, @0, @0, @0, @0, @1, @0, @0, @0, @0, @0,
                                           @0, @0, @0, @1, @1, @1, @1, @1, @1, @1, @1, @1]];
        
    }
}];

[_krBPN trainingByRandomSettings];
```

#### Sample 4

To forecast survival rate in medical of cancer. (癌症醫療存活率預測)

``` objective-c
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

__block typeof(_krBPN) _weakKrBPN = _krBPN
[_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
    if( success )
    {
        NSLog(@"Training done with total times : %i", totalTimes);
        NSLog(@"TrainedInfo : %@", trainedInfo);
        
        //Start in checking the network is correctly trained.
        NSLog(@"======== Start in Verification ========");

        [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"Verified TrainedInfo : %@", trainedInfo);
        }];
        
        [_weakKrBPN recoverNetwork];

        // To predict how many years could be alive with the patients.
        [_weakKrBPN directOutputAtInputs:@[@0.6, @0.6, @0.0, @0.25, @0.0, @0.1f]];
        [_weakKrBPN directOutputAtInputs:@[@0.6, @0.6, @0.0, @0.25, @1.0, @0.1f]];
    }
}];

[_krBPN trainingByRandomSettings];
```

## Version

V2.0.6

## License

MIT.
