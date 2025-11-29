# MQL5 Book - Part 4 (Pages 601-800)

## Page 601

Part 4. Common APIs
601 
4.1 0 Matrices and vectors
Virtual balance of trading a portfolio of currencies by lots according to the decision
The "future" part looks less smooth, and we can even say that we are lucky that it continues to grow,
despite such a simple model. As a rule, during the forward test, the virtual balance curve significantly
degrades and starts to go down.
It is important to note that to test the model, we took the obtained X values from the "as is" solution of
the system, while in practice we will need to normalize them to the minimum lots and lot step, which
will negatively affect the results and bring them closer to reality.
4.1 0.1 3 Machine learning methods
Among the built-in methods of matrices and vectors, there are several that are in demand in machine
learning tasks, in particular, in the implementation of neural networks.
As the name implies, a neural network is a collection of many neurons which are primitive computing
cells. They are primitive in the sense that they perform fairly simple calculations: as a rule, a neuron
has a set of weight coefficients that are applied to certain input signals, after which the weighted sum
of the signals is fed into the function, which is a nonlinear converter.
The use of an activation function amplifies weak signals and limits those that are too strong, preventing
the transition to saturation (overflow of real calculations). However, the most important thing is that
nonlinearity gives the network new computing capabilities, enabling the solution of more complicated
problems.
Elementary neural network
The power of neural networks is manifested by combining a large number of neurons and establishing
connections between them. Usually, neurons are organized into layers (which can be compared with
matrices or vectors), including those with recursive (recurrent) connections, and can also have
activation functions that differ in their effect. This makes it possible to analyze volumetric data using
various algorithms, in particular, by finding hidden patterns in them.
Note that if it were not for the non-linearity in each neuron, a multilayer neural network could be
represented in equivalent form as a single layer, whose coefficients are obtained by the matrix product

---

## Page 602

Part 4. Common APIs
602
4.1 0 Matrices and vectors
of all layers (Wtotal = W1 * W2 * ... * WL, where 1 ..L are the numbers of layers). And this would be a simple
linear adder. Thus, the importance of activation functions is mathematically substantiated.
Some of the most famous activation functions
One of the main classifications of neural networks divides them according to the learning algorithm
used into supervised and unsupervised learning networks. Supervised ones require a human expert to
provide the desired outputs for the original data set (for example, discrete markers of the state of a
trading system, or numerical indicators of implied price increments). Unsupervised networks identify
clusters in the data on their own.
In any case, the task of training a neural network is to find parameters that minimize the error on the
training and test samples, for which the loss function is used: it provides a qualitative or quantitative
estimate of the error between the target and the received network response.
The most important aspects for the successful application of neural networks include the selection
of informative and mutually independent predictors (analyzed characteristics), data transformation
(normalization and cleaning) according to the specifics of the learning algorithm, as well as network
architecture and size optimization. Please note that the use of machine learning algorithms does
not guarantee success.
Here, we will not go into the theory of neural networks, their classification, and typical tasks to be
solved. This topic is too broad. Those interested can find articles on the mql5.com website and in other
sources.
MQL5 provides three machine learning methods which have become part of the matrix and vector API.
• Activation calculates the values of the activation function
• Derivative calculates the values of the derivative of the activation function
• Loss  calculates the value of the loss function

---

## Page 603

Part 4. Common APIs
603
4.1 0 Matrices and vectors
Derivatives of activation functions enable the efficient update of model parameters based on the model
error which changes during the learning process.
The first two methods write the result to the passed vector/matrix and return a success indicator (true
or false), and the loss function returns a number. Let's present their prototypes (under the type
obj ect<T> we marked both, matrix<T> and vector<T>):
bool object<T>::Activation(object<T> &out, ENUM_ACTIVATION_FUNCTION activation)
bool object<T>::Derivative(object<T> &out, ENUM_ACTIVATION_FUNCTION loss)
T object<T>::Loss(const object<T> &target, ENUM_LOSS_FUNCTION loss)
Some activation functions allow setting a parameter with a third, optional argument.
Please refer to the MQL5 Documentation for the list of supported activation functions in the
ENUM_ACTIVATION_FUNCTION enumeration and loss functions in the ENUM_LOSS_FUNCTION
enumeration.
As an introductory example, let's consider the problem of analyzing the real tick stream. Some traders
consider ticks to be garbage noise, while others practice tick-based high-frequency trading. There is an
assumption that high-frequency algorithms, as a rule, give an advantage to big players and are based
solely on the software processing of price information. Based on this, we will put forward a hypothesis
that there is a short-term memory effect in the tick stream, due to the market makers' currently
active robots. Then, a machine learning method can be used to find this dependence and to predict
several future ticks.
Machine learning always involves putting forward hypotheses, synthesizing a model for them, and
testing them in practice. Obviously, productive hypotheses are not always obtained. It is a long
process of trial and error, in which failure is a source of improvement and new ideas.
We will use one of the simplest types of neural networks: Bidirectional Associative Memory (BAM). Such
a network has only two layers: input and output. A certain response (association) is formed in the
output in response to the input signal. Layer sizes may vary. When the sizes are the same, the result is
a Hopfield network.


---

## Page 604

Part 4. Common APIs
604
4.1 0 Matrices and vectors
Fully connected Bidirectional Associative Memory
Using such a network, we will compare N recent previous ticks and M next predicted ticks, forming a
training sample from the near past to a given depth. Ticks will be fed into the network as positive or
negative price increments converted to binary values [+1 , -1 ] (binary signals are the canonical form of
coding in BAM and Hopfield networks).
The most important advantage of BAM is the almost instantaneous (compared to most other iterative
methods) learning process, which consists in calculating the weight matrix. We will give the formula
below.
However, this simplicity also has a downside: the BAM capacity (the number of images that it can
remember) is limited to the smallest layer size, provided that the condition of a special distribution of
+1  and -1  in the training sample vectors is met.
Thus, for our case, the network will generalize all sequences of ticks in the training sample and then, in
the course of regular work, it will roll down to one or another stored image, depending on the sequence
of new ticks presented. How well this will turn out in practice depends on a very large number of
factors, including the network size and settings, the characteristics of the current tick stream, and
others.
Because it is assumed that the tick stream has only short-term memory, it is desirable to retrain the
network in real time or close to it, since training is actually reduced to several matrix operations.
So, in order for the network to remember the associative images (in our case, the past and the future
of the tick stream), the following equation is required:
W = Σi(Ai
TBi)
where W is the weight matrix of the network. The summation is performed over all pairwise products of
the input vectors Ai and corresponding output vectors Bi.
Then, when the network is running, we feed the input image to the first layer, apply the W matrix to it,
and thereby activate the second layer, in which the activation function for each neuron is calculated.
After that, using the transposed W T matrix, the signal propagates back to the first layer, where
activation functions are also applied in neurons. At this moment, the input image no longer arrives at
the first layer, i.e., the free oscillatory process continues in the network. It continues until the changes
in the signal of the network neurons stabilize (i.e., become less than a certain predetermined value).
In this state, the second layer of the network contains the found associated output image – the
prediction.
Let's implement this machine learning scenario in the script MatrixMachineLearning.mq5.
In the input parameters, you can set the total number of last ticks (TicksToLoad) requested from the
history, and how many of them are allocated for testing (TicksToTest). Accordingly, the model
(weights) will be based on (TicksToLoad - TicksToTest) ticks.
input int TicksToLoad = 100;
input int TicksToTest = 50;
input int PredictorSize = 20;
input int ForecastSize = 10;
Also, in the input variables, the sizes of the input vector (the number of known ticks PredictorSize) and
output vector (the number of future ticks ForecastSize) are selected.

---

## Page 605

Part 4. Common APIs
605
4.1 0 Matrices and vectors
Ticks are requested at the beginning of the OnStart function. In this case, we only work with Ask
prices. However, you can also add Bid and Last process, along with volumes.
void OnStart()
{
   vector ticks;
   ticks.CopyTicks(_Symbol, COPY_TICKS_ALL | COPY_TICKS_ASK, 0, TicksToLoad);
   ...
Let's split ticks into training and test sets.
   vector ask1(n - TicksToTest);
   for(int i = 0; i < n - TicksToTest; ++i)
   {
      ask1[i] = ticks[i];
   }
   
   vector ask2(TicksToTest);
   for(int i = 0; i < TicksToTest; ++i)
   {
      ask2[i] = ticks[i + TicksToLoad - TicksToTest];
   }
   ...
To calculate price increments, we use the Convolve method with an additional vector {+1 , -1 }. Note
that the vector with increments will be 1  element shorter than the original.
   vector differentiator = {+1, -1};
   vector deltas = ask1.Convolve(differentiator, VECTOR_CONVOLVE_VALID);
   ...
Convolution according to the VECTOR_CONVOLVE_VALID algorithm means that only full overlaps of
vectors are taken into account (i.e., the smaller vector is sequentially shifted along the larger one
without moving beyond its boundaries). Other types of convolutions allow vectors to overlap with only
one element, or half of the elements (in this case, the remaining elements are beyond the
corresponding vector and the convolution values show border effects).
To convert continuous values of increments into unit pulses (positive and negative depending on the
sign of the initial element of the vector), we will use an auxiliary function Binary (not shown here): it
returns a new copy of the vector in which each element is either +1  or -1 .
   vector inputs = Binary(deltas);
Based on the received input sequence, we use the TrainWeights function to calculate the W neural
network weight matrix. We will consider the structure of this function later. For now, please pay
attention that the PredictorSize and ForecastSize parameters are passed to it, which enables the
splitting of a continuous sequence of ticks into sets of paired input and output vectors according to the
size of the input and output BAM layers, respectively.

---

## Page 606

Part 4. Common APIs
606
4.1 0 Matrices and vectors
   matrix W = TrainWeights(inputs, PredictorSize, ForecastSize);
   Print("Check training on backtest: ");   
   CheckWeights(W, inputs);
   ...
Immediately after training the network, we check its accuracy on the training set: just to make sure
that the network has been trained. This is implemented by the CheckWeights function.
However, it is more important to check how the network behaves on unfamiliar test data. To do this,
let's differentiate and binarize the second vector ask2 and then also send it to CheckWeights.
   vector test = Binary(ask2.Convolve(differentiator, VECTOR_CONVOLVE_VALID));
   Print("Check training on forwardtest: ");   
   CheckWeights(W, test);
   ...
}
It's time to get acquainted with the TrainWeights function, in which we define A and B matrices to
"slice" vectors from the passed input sequence, i.e. from the data vector.
template<typename T>
matrix<T> TrainWeights(const vector<T> &data, const uint predictor, const uint responce, 
   const uint start = 0, const uint _stop = 0, const uint step = 1)
{
   const uint sample = predictor + responce;
   const uint stop = _stop <= start ? (uint)data.Size() : _stop;
   const uint n = (stop - sample + 1 - start) / step;
   matrix<T> A(n, predictor), B(n, responce);
   
   ulong k = 0;
   for(ulong i = start; i < stop - sample + 1; i += step, ++k)
   {
      for(ulong j = 0; j < predictor; ++j)
      {
         A[k][j] = data[start + i * step + j];
      }
      for(ulong j = 0; j < responce; ++j)
      {
         B[k][j] = data[start + i * step + j + predictor];
      }
   }
   ...
Each successive A pattern is obtained from consecutive ticks in quantity equal to predictor, and the
future pattern corresponding to is obtained from the following response elements. As long as the total
amount of data allows, this window shifts to the right, one element at a time, forming more new pairs of
images. Images are numbered by rows, and ticks in them are numbered by columns.
Next, we should allocate memory for the weight matrix W and fill it using matrix methods: we
sequentially multiply rows from A and B using Outer, and then perform matrix summation.

---

## Page 607

Part 4. Common APIs
607
4.1 0 Matrices and vectors
   matrix<T> W = matrix<T>::Zeros(predictor, responce);
   
   for(ulong i = 0; i < k; ++i)
   {
      W += A.Row(i).Outer(B.Row(i));
   }
   
   return W;
}
The CheckWeights function performs similar actions for a neural network, the weight coefficients of
which are passed in a ready-made form in the first W argument. The sizes of the training vectors are
extracted from the W matrix itself.
template<typename T>
void CheckWeights(const matrix<T> &W, 
   const vector<T> &data, 
   const uint start = 0, const uint _stop = 0, const uint step = 1)
{
   const uint predictor = (uint)W.Rows();
   const uint responce = (uint)W.Cols();
   const uint sample = predictor + responce;
   const uint stop = _stop <= start ? (uint)data.Size() : _stop;
   const uint n = (stop - sample + 1 - start) / step;
   matrix<T> A(n, predictor), B(n, responce);
   
   ulong k = 0;
   for(ulong i = start; i < stop - sample + 1; i += step, ++k)
   {
      for(ulong j = 0; j < predictor; ++j)
      {
         A[k][j] = data[start + i * step + j];
      }
      for(ulong j = 0; j < responce; ++j)
      {
         B[k][j] = data[start + i * step + j + predictor];
      }
   }
   
   const matrix<T> w = W.Transpose();
   ...
Matrices A and B in this case are not formed to calculate W but act as "suppliers" of vectors for
testing. We also need a transposed copy of W to calculate the return signals from the second network
layer to the first.
The number of iterations during which transient processes are allowed in the network, up to
convergence, is limited by the limit constant.

---

## Page 608

Part 4. Common APIs
608
4.1 0 Matrices and vectors
   const uint limit = 100;
   
   int positive = 0;
   int negative = 0;
   int average = 0;
Variables positive, negative, and average are needed to calculate the statistics of successful and
unsuccessful predictions in order to evaluate the quality of training.
Further, the network is activated in a loop over test pattern pairs and its final response is taken. Each
next input vector is written into vector a, and output layer b is filled with zeros. After that, iterations
are launched for signal transmission from a to b using the matrix W and applying the activation function
AF_TANH, as well as for the feedback signal from b to a, and also the use of AF_TANH. The process
continues until reaching limit loops (which is unlikely) or until the convergence condition is fulfilled,
under which the a and b neuron state vectors practically do not change (here we use the Compare
method and auxiliary copies of x and y vectors from the previous iteration).
   for(ulong i = 0; i < k; ++i)
   {
      vector a = A.Row(i);
      vector b = vector::Zeros(responce);
      vector x, y;
      uint j = 0;
      
      for( ; j < limit; ++j)
      {
         x = a;
         y = b;
         a.MatMul(W).Activation(b, AF_TANH);
         b.MatMul(w).Activation(a, AF_TANH);
         if(!a.Compare(x, 0.00001) && !b.Compare(y, 0.00001)) break;
      }
      
      Binarize(a);
      Binarize(b);
      ...
After reaching a stable state, we transfer the states of neurons from continuous (real) to binary +1  and
-1  using the Binarize function (it is similar to the previously mentioned Binary function, but changes the
state of the vector in place).
Now, we only need to count the number of matches in the output layer with the target vector. For this,
perform scalar multiplication of vectors. A positive result means that the number of correctly guessed
ticks exceeds the number of incorrect ones. The total hit count is accumulated in 'average'.
      const int match = (int)(b.Dot(B.Row(i)));
      if(match > 0) positive++;
      else if(match < 0) negative++;
      
      average += match; // 0 in match means 50/50 precision (i.e. random guessing)
   }
After the cycle is completed for all test samples, we display statistics.

---

## Page 609

Part 4. Common APIs
609
4.1 0 Matrices and vectors
   float skew = (float)average / k; // average number of matches per vector
   
   PrintFormat("Count=%d Positive=%d Negative=%d Accuracy=%.2f%%", 
      k, positive, negative, ((skew + responce) / 2 / responce) * 100);
}
The script also includes the RunWeights function which represents a working run of the neural network
(by its weight matrix W) for the online vector from the last predictor ticks. The function will return a
vector with estimated future ticks.
template<typename T>
vector<T> RunWeights(const matrix<T> &W, const vector<T> &data)
{
   const uint predictor = (uint)W.Rows();
   const uint responce = (uint)W.Cols();
   vector a = data;
   vector b = vector::Zeros(responce);
   
   vector x, y;
   uint j = 0;
   const uint limit = LIMIT;
   const matrix<T> w = W.Transpose();
   
   for( ; j < limit; ++j)
   {
      x = a;
      y = b;
      a.MatMul(W).Activation(b, AF_TANH);
      b.MatMul(w).Activation(a, AF_TANH);
      if(!a.Compare(x, 0.00001) && !b.Compare(y, 0.00001)) break;
   }
   
   Binarize(b);
   
   return b;
}
At the end of OnStart, we pause execution for 1  second (in order to wait for new ticks with a certain
degree of probability), request the last PredictorSize + 1  ticks (do not forget +1  for differentiation), and
make predictions for them online.

---

## Page 610

Part 4. Common APIs
61 0
4.1 0 Matrices and vectors
void OnStart()
{
   ...
   Sleep(1000);
   vector ask3;
   ask3.CopyTicks(_Symbol, COPY_TICKS_ALL | COPY_TICKS_ASK, 0, PredictorSize + 1);
   vector online = Binary(ask3.Convolve(differentiator, VECTOR_CONVOLVE_VALID));
   Print("Online: ", online);
   vector forecast = RunWeights(W, online);
   Print("Forecast: ", forecast);
}
Running the script with the default settings on EURUSD on Friday evening gave the following results.
Check training on backtest: 
Count=20 Positive=20 Negative=0 Accuracy=85.50%
Check training on forwardtest: 
Count=20 Positive=12 Negative=2 Accuracy=58.50%
Online: [1,1,1,1,-1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1]
Forecast: [-1,1,-1,1,-1,-1,1,1,-1,1]
The symbol and time are not mentioned since the market situation can significantly affect the
applicability of the algorithm and the specific network configuration. When the market is open, every
time you run the script you will get new results as more and more ticks come in. This is an expected
behavior consistent with the short memory formation hypothesis.
As we can see, the training accuracy is acceptable, but it noticeably decreases on test data and may
fall below 50%.
At this point, we smoothly move from programming to the field of scientific research. The machine
learning toolkit built into MQL5 allows you to implement many other configurations of neural networks
and analyzers, with different trading strategies and principles for preparing initial data.

---

## Page 611

Part 5. Creating application programs
61 1 
 
Part 5. Creating application programs in MQL5
In this part, we will closely study those sections of the API that are related to solving applied problems
of algorithmic trading: analysis and processing of financial data, their visualization and markup using
graphic objects, automation of routine actions, and interactive user interaction.
Let's start with the general principles of creating MQL programs, their types, features, and the event
model in the terminal. Then we will touch on access to timeseries, work with charts and graphical
objects. Finally, let's analyze the principles of creating and using each type of MQL program separately.
Active users of MetaTrader 5 undoubtedly remember that the terminal supports five types of programs:
• Technical indicators for calculating arbitrary indicators in the form of time series, with the
possibility of their visualization in the main chart window, or in a separate panel (sub-window);
• Expert Advisors providing automatic or semi-automatic trading;
• Scripts for performing auxiliary one-time tasks on demand;
• Services for performing background tasks in continuous mode;
• Libraries, which are compiled modules with a specific, separate functionality, which are connected
to other types of MQL programs during their loading, dynamically (which fundamentally
distinguishes libraries from header files that are included statically at the compilation stage).
In the previous parts of the book, as we mastered the basics of programming and common built-in
functions, we already had to turn to the implementation of scripts and services as examples. These
types of programs were chosen as being simpler than the others. Now we will describe them
purposefully and add more functional and popular indicators to them.
With the help of indicators and charts, we will learn some techniques that will be applicable to Expert
Advisors as well. However, we will postpone the actual development of Expert Advisors, which is a more
complex task in its essence, and move it to a separate, following Part 6, which includes not only
automatic execution of orders and formalization of trading strategies, but also their backtesting and
optimization.
As far as indicators are concerned, MetaTrader 5 is known to come with a set of built-in standard
indicators. In this part, we will learn how to use them programmatically, as well as create our own
indicators both from scratch, and based on other indicators.
All compiled indicators, Expert Advisors, scripts and services are displayed in the Navigator in
MetaTrader 5. Libraries are not independent programs, and therefore do not have a dedicated branch
in the hierarchy, although, of course, this would be convenient from the point of view of uniform
management of all binary modules. As we will see later, those programs that depend on a particular
library cannot run without it. But now you can check the existence of the library only in the file
manager.
MQL5 Programming for Traders – Source Codes from the Book. Part 5
Examples from the book are also available in the public project \MQL5\Shared Projects\MQL5Book

---

## Page 612

Part 5. Creating application programs
61 2
5.1  General principles for executing MQL programs
5.1  General principles for executing MQL programs
All MQL programs can be broadly divided into several groups depending on their capabilities and
features.
Most programs, such as Expert Advisors, indicators, and scripts, work in the context of a chart. In other
words, they start executing only after they are attached to one of the open charts by using the Attach
to Chart context menu command in the Navigator tree or by dragging and dropping from Navigator to
the chart.
In contrast, services cannot be placed on the chart, as they are designed to perform long, cyclic
actions in the background. For example, in a service, you can create a custom symbol and then receive
its data and keep updating it in an endless loop using network functions. Another logical application of a
service is monitoring the trading account and the network connection, as a part of a solution that
notifies the user about communication problems.
It is important to note that indicators and Expert Advisors are saved on the chart between terminal
working sessions. In other words, if, for example, a user runs an indicator on the chart and then,
without explicitly deleting it, closes MetaTrader 5, then the next time the terminal starts, the indicator
will be restored along with the chart, including all its settings.
By the way, linking indicators and Expert Advisors to the chart is the basis for templates (see the
Documentation). The user can create a set of programs to be used on a chart, configure them and save
the set in a special file with the tpl extension. This is done using the context menu command Templates
-> Save. After that, you can apply the template to any new chart (command Templates -> Upload) and
run all linked programs. Templates are stored in the directory MQL5/Profiles/Templates/ by default.
Another consequence of attaching to a chart is that closing a chart results in unloading all MQL
programs that were placed on it. However, MetaTrader 5 saves all closed charts in a specific way (at
least for a while) and therefore, if the chart was closed by accident, it can be restored along with all
programs (and graphic objects) using the command File -> Open Remote.
If for some reason the terminal fails to load chart files, the entire state of MQL programs (settings and
location) will be lost. Basically, the same applies to graphic objects – programs can add them for their
own needs and expect that these objects are located on the chart. Make backup copies of charts. Each
chart is a file with the extension chr. Such files are stored by default in the directory
MQL5/Profiles/Charts/Default/. This is the standard profile created when the platform is installed. You
can create other profiles with the menu command File -> Profiles and then switch between them (see
the Documentation).
If necessary, you can stop an Expert Advisor and remove it from the chart using the context menu
command Expert list (called by pressing the right mouse button in the chart window). It opens the
Experts dialog with a list of all Expert Advisors running in the terminal. In this list, select an Expert
Advisor that you no longer need and press Remove.
Indicators can also be removed explicitly, using a similar context menu command Indicator List. It
opens a dialog with a list of indicators running on the current chart, in which you can select a specific
indicator and click the button Remove. In addition, most indicators display various graphical
constructions, such as lines and histograms, on the chart, which can also be deleted using the relevant
context menu commands.
In contrast to indicators and Expert Advisors, scripts are not permanently attached to a chart. In
standard mode, the script is removed from the chart automatically after the task assigned to it is

---

## Page 613

Part 5. Creating application programs
61 3
5.1  General principles for executing MQL programs
completed, if this is a one-time action. If a script has a loop for periodic, repetitive actions, it will, of
course, continue its work until the loop is interrupted in one way or another, but no longer than until
the end of the session. Closing the terminal causes the script to become detached from the chart. After
restarting MetaTrader 5, scripts are not restored on charts.
Please note that if you switch the chart to another symbol or timeframe, the script running on it will
be unloaded. But indicators and Expert Advisors will continue to work, however, they will be re-
initialized. Initialization rules for them are different. These details will be discussed in the section
Features of starting and stopping programs of various types.
Only one Expert Advisor, only one script, and any number of indicators can be placed on the chart. The
Expert Advisor, the script, and all indicators will work in parallel (simultaneously).
As for services, their created and running instances are automatically restored after loading the
terminal. The service instance can be stopped or deleted using the context menu in the Services
section of the Navigator window.
The following table summarizes the properties described above in a summary form.
Program type
Link to chart
Quantity
on the chart
Recovery of the 
session
Indicator
Required
Multiple
With chart or template
Expert Advisor
Required
Maximum 1 
With chart or template
Script
Required
Maximum 1 
Not supported
Service
Not supported
0
With terminal
All MQL programs are executed in the client terminal and therefore work only while the terminal is
open. For constant program control over the account, use a VPS.
5.1 .1  Designing MQL programs of various types
The program type is a fundamental property in MQL5. In contrast to C++ or other general-purpose
programming languages, where any program can be developed in arbitrary directions, for example, by
adding a graphical interface or uploading data from a server over the network, MQL programs are
divided into certain groups according to their purpose. For example, technical timeseries analysis with
visualization is implemented via indicators, but they are not able to trade. In turn, the trading API
functions are available to Expert Advisors, but they lack indicator buffers (arrays for drawing lines).
Therefore, when solving a specific applied problem, the developer should decompose it into parts, and
the functionality of each part should fit into the specialization of a separate type. Of course, in simple
cases, a single MQL program is enough, but sometimes the optimal technical solution is not obvious.
For example, how would you implement the plotting of a Renko chart: as an indicator, as a custom
symbol generated by the service, or can as specific calculations directly in the trading Expert Advisor?
All options are possible.
The type of MQL program is characterized by several factors.
First, each type of program has a separate folder in the MQL5 working directory. We have already
mentioned this fact in the introduction to Part 1  and listed the folders. So, for indicators, Expert

---

## Page 614

Part 5. Creating application programs
61 4
5.1  General principles for executing MQL programs
Advisors, scripts, and services, the designated folders are Indicators, Experts, Scripts, and Services,
respectively. The Libraries subfolder is reserved for libraries in the MQL5 folder. In each of them, you
can organize a tree of nested folders of arbitrary configuration.
The binary file (the finished program with the extension ex5) – which is a result of compiling the mq5
file – is generated in the same directory as the source mq5 file. However, we should also mention
projects in MetaEditor (files with the extension mqproj ), which we will analyze in the chapter Projects.
When a project is developed, a finished product is created in a directory next to the project. When
creating a program from the MQL5 Wizard in MetaEditor (command File -> New), the source file is
placed by default in the folder corresponding to the program type. If you accidentally copy a program
to the wrong directory, nothing terrible will happen: it will not turn, for example, from an Expert Advisor
into an indicator, or vice versa. It can be moved to the desired location directly in the editor, inside the
Navigator window, or in an external file manager. In the Navigator, each program type is displayed with
a special icon.
The location of a program within the MQL5 directory in a subfolder dedicated to a particular type
does not determine the type of this particular MQL program. The type is determined based on the
contents of the executable file, which, in turn, is formed by the compiler from property directives
and statements in the source code. 
The hierarchy of folders by program types is used for convenience. It is recommended to stick to it,
except when it comes to a group of related projects (with programs of different types), which are
more logical to store in a separate directory.
Second, each type of program is characterized by support for a limited, specific set of system events
that activate the program. We will see an Overview of event-handling functions in a separate section.
To receive events of a specific type in a program, it is necessary to describe a handler function with a
predefined prototype (name, list of parameters, return value).
For example, we have already seen that in scripts and services, work is started in the OnStart function,
and since it is the only one there, it can be called the main "entry point" through which the terminal
transfers control to the application code. In other types of programs, the situation is somewhat more
complicated. In general, we note that a program type is characterized by a certain set of handlers,
some of which may be mandatory and some are optional (but at the same time, unacceptable for other
types of programs). In particular, an indicator requires the OnCalculate function (without it, an indicator
will not compile and the compiler will generate an error). However, this function is not used in Expert
Advisors.
Third, some types of programs require special #property directives. In the chapter General properties
of programs, we have already seen directives that can be used in all types of programs. However, there
are other, specialized directives. For example, in tasks with services, that we mentioned, we met the
#property service directive, which makes the program a service. Without it, even placing the program in
the Services folder will not allow it to run in the background.
Similarly, the #property library directive plays a defining role in the creation of libraries. All such
directive properties will be discussed in the sections for the corresponding types of programs.
The combination of directives and event handlers is taken into account when establishing an MQL
program type in the following order (top to bottom until the first match):
·indicator: the presence of the OnCalculate handler
·library: #property library
·script: the presence of the OnStart  handler and the absence of #property service

---

## Page 615

Part 5. Creating application programs
61 5
5.1  General principles for executing MQL programs
·service: the presence of the OnStart handler and #property service
·Expert Advisor: the presence of any other handler
An example of what effect these properties have on the compiler will be given in the section Overview of
event handling functions.
For all of the above points, one more point should be taken into account. The program type is
determined by the main compiled module: a file with the mq5 extension, where other sources from
other directories can be included using the #include directive. All functions included in this way are
taken into consideration on the same level as those that are present directly in the main mq5 file. 
On the other hand, #property directives have an effect only when placed in the main compiled mq5
file. If the directives occur in files included in the program using #include, they will be ignored.
The main mq5 file does not have to literally contain event handler functions. It is perfectly acceptable
to place part or all of the algorithm in mqh header files and then include them in one or more programs.
For example, we can implement the OnStart handler with a set of useful actions in an mqh file and use
it via #include inside two separate programs: a script and a service.
Meanwhile, let's note that the presence of common event handlers is not the only motive for separating
common algorithm fragments into a header file. You can use the same calculation formula, for example,
in an indicator and in an Expert Advisor, leaving their event handlers in the main program modules.
Although it is customary to refer to include files as header files and give them the mqh extension,
this is not technically necessary. It is quite acceptable (although not recommended) to include
another mq5 file or, for example, a txt file in one mq5 file. They may contain some legacy code or,
let's say, initialization of certain arrays with constants. The inclusion of another mq5 file does not
make it the main one. 
You should make sure that only the event-handling functions characteristic of the specific program
type get into the program, and that there are no duplicates among them (as you know, functions
are identified by a combination of names and a list of parameters: function overload only allowed
with a different set of parameters). This is usually achieved using various preprocessor directives.
For example, by defining the macro #define OnStart OnStartPrevious before including a third-party
mq5 script file in some of our programs, we will actually turn the OnStart function described in it
into OnStartPrevious, and we can call it as usual from our own event handlers. 
However, this approach makes sense only in exceptional cases when the source code of the
included mq5 file cannot be modified due to some reason, in particular, when it cannot be
structured with the selection of algorithms of interest into functions or classes in separate header
files.
According to the principle of interaction with the user, MQL programs can be divided into interactive
and utilitarian ones.
Interactive programs – indicators and Expert Advisors – can process events, which occur in the
software environment in response to user actions, such as pressing buttons on the keyboard, moving
the mouse, changing the window size, as well as many other events, for example, related to receiving
quote data or to timer actions.
Utility programs – services and scripts – are guided only by input variables set at the time of launch,
and do not respond to events in the system.
Apart from all types of programs are libraries. They are always executed as part of another type of
MQL program (one of the four main ones), and therefore do not have any distinctive characteristics or

---

## Page 616

Part 5. Creating application programs
61 6
5.1  General principles for executing MQL programs
behavior. In particular, they cannot directly receive events from the terminal and do not have their own
threads (see next section). The same library can be connected to many programs, and this happens
dynamically at the time of the launch of each parent program. In the section on libraries, we'll learn
how to describe a library's exported API and import it into a parent program.
5.1 .2 Threads
In a simplified form, a program can be represented as a sequence of statements that a developer has
generated for a computer. The main executor of statements in a computer is the central processing
unit. Modern computers are usually equipped with processors with multiple cores, which is equivalent to
having multiple processors. However, the number of programs a user may want to run in parallel is
virtually unlimited. Thus, the number of programs is always many times greater than the available
cores/processors. Due to this, each core actually divides its working time between several different
programs: it will allocate 1  millisecond for executing the statements of one program, then 1  millisecond
for the statements of another, then for thirds, and so on, in a circle. Since the switching occurs very
quickly, the user does not notice this, as it seems that all programs are executed in parallel and
simultaneously.
For the processor to be able to suspend the execution of the statements of one program and then
resume its work from the previous place (after it quietly switched to the statements of other "parallel"
programs), it must be able to somehow save and restore the intermediate state of each program: the
current statement, variables, possibly open files, network connections, and so on. This entire collection
of resources and data that a program needs to run normally, along with its current position in the
sequence of statements, is called the program's execution context. The operating system, in fact, is
designed to create such contexts for each program at the request of the user (or other programs).
Each such active context is called a thread. Many programs require many threads for themselves
because their functionality involves maintaining several activities in parallel. MetaTrader 5 also requires
multiple threads to load loading quotes for multiple symbols, plot charts, and respond to user actions.
Furthermore, separate threads are also allocated to MQL programs.
The MQL program execution environment allocates no more than one thread to each program. Expert
Advisors, scripts, and services receive strictly one thread each. As for indicators, one stream is
allocated for all indicators working on one financial instrument. Moreover, the same thread is
responsible for displaying the charts of the corresponding symbol, so it is not recommended to occupy
it with heavy calculations. Otherwise, the user interface will become unresponsive: user actions will be
processed with a delay, or the window will even become unresponsive. Threads of all other types of
MQL programs are not tied to an interface and, therefore, can load the processor with any complex
task.
One of the important properties of a thread follows from its definition and purpose: It only supports
sequential execution of specified statements one after another. Only one statement is executed in one
thread at a time. If an infinite loop is written in the program, the thread will get stuck on this
instruction and never get to the instructions below it. Long calculations can also create the effect of an
endless loop: they will load the processor and prevent other actions from being performed, the results
of which the user may expect. That is why efficient calculations in indicators are important for the
smooth operation of the graphical interface.
However, in other types of MQL programs, attention should be paid to thread arrangement. In the
following sections, we will get familiar with the special event handling functions that are the entry points
to MQL programs. A single-threaded model means that during the processing of one event, the program
is immune to other events that could potentially occur at the same time. Therefore, the terminal

---

## Page 617

Part 5. Creating application programs
61 7
5.1  General principles for executing MQL programs
organizes an event queue for each program. We will touch on this point in more detail in the next
section.
In order to experience the effects of single-threading in practice, we will look at a simple example in the
section Limitations and benefits of indicators (IndBarIndex.mq5). We have chosen indicators for this
purpose because they not only share one thread for each symbol but also display results directly on the
chart, which makes the potential problem the most obvious.
5.1 .3 Overview of event handling functions
The transfer of control to MQL programs, that is, their execution, occurs by calling special functions by
the terminal or test agents, which the MQL developer defines in their application code to process
predefined events. Such functions must have a specified prototype, including a name, a list of
parameters (number, types, and order), and a return type.
The name of each function corresponds to the meaning of the event, with the addition of the prefix On.
For example, OnStart is the main function for "starting" scripts and services; it is called by the terminal
at the moment the script is placed on the chart or the service instance is launched.
For the purposes of this book, we will refer to an event and its corresponding handler by the same
name.
The following table lists all event types and programs that support them (
– indicator, 
 – Expert
Advisor, 
 – script, 
 – service). A detailed description of the events is given in the sections of the
respective program types. Many factors can cause initialization and deinitialization events: placing the
program on the chart, changing its settings, changing the symbol/timeframe of the chart (or template,
or profile), changing the account, and others (see chapter Features of starting and stopping programs
of various types).

---

## Page 618

Part 5. Creating application programs
61 8
5.1  General principles for executing MQL programs
Program type
Event/Handler
Description
OnStart
-
-
●
●
Start/Execute
OnInit
+
+
-
-
Initialization after loading (see details in
section Features of starting and stopping
programs of various types)
OnDeinit
+
+
-
-
Deinitialization before stopping and
unloading
OnTick
-
+
-
-
Getting a new price (tick)
OnCalculate
●
-
-
-
Request to recalculate the indicator due to
receiving a new price or synchronizing old
prices
OnTimer
+
+
-
-
Timer activation with a specified frequency
OnTrade
-
+
-
-
Completion of a trading operation on the
server
OnTradeTransa
ction
-
+
-
-
Changing the state of the trading account
(orders, deals, positions)
OnBookEvent
+
+
-
-
Change in the order book
OnChartEvent
+
+
-
-
User or MQL program action on the chart
OnTester
-
+
-
-
End of a single tester pass
OnTesterInit
-
+
-
-
Initialization before optimization
OnTesterDeinit
-
+
-
-
Deinitialization after optimization
OnTesterPass
-
+
-
-
Receiving optimization data from the testing
agent
Mandatory handlers are marked with symbol '●', and optional handlers are marked with '+'.
Although handler functions are primarily intended to be called by the runtime, you can also call them
from your own source code. For example, if an Expert Advisor needs to make some calculation based on
the available quotes immediately after the start, and even in the absence of ticks (for example, on
weekends), you can call OnTick before leaving OnInit. Alternatively, it would be logical to separate the
calculation into a separate function and call it both from OnInit and from OnTick. However, it is
desirable to perform the work of the initialization function quickly, and if the calculation is long, it
should be performed on a timer.

---

## Page 619

Part 5. Creating application programs
61 9
5.1  General principles for executing MQL programs
All MQL programs (except libraries) must have at least one event handler. Otherwise, the compiler will
generate an "event handling function not found" error.
The presence of some handler functions determines the type of the program in the absence of
#property directives that set another type. For example, having the OnCalculate handler leads to the
generation of the indicator (even if it is located in another folder, for example, scripts or Expert
Advisors). The presence of the OnStart handler (if there is no OnCalculate) means creating a script. At
the same time, if the indicator, in addition to OnCalculate, will face OnStart, we get a compiler warning
"OnStart function defined in the non-script program".
The book includes two files: AllInOne.mq5 and AllInOne.mqh. The header file describes almost empty
templates of all the main event handlers. They contain nothing except outputting the name of the
handler to the log. We will consider the syntax and specifics of using each of the handlers in the
sections on specific types of MQL programs. The meaning of this file is to provide a field for
experiments with compiling different types of programs, depending on the presence of certain handlers
and property directives (#property).
Some combinations may result in errors or warnings.
If the compilation was successful, then the resulting program type is automatically logged after it is
loaded using the following line:
const string type = 
   PRTF(EnumToString((ENUM_PROGRAM_TYPE)MQLInfoInteger(MQL_PROGRAM_TYPE)));
We studied the enum ENUM_PROGRAM_TYPE and function MQLInfoInteger in the section Program type
and license.
The file AllInOne.mq5, which includes AllInOne.mqh, is initially located in the directory
MQL5Book/Scripts/p5/, but it can be copied to any other folder, including neighboring Navigator
branches (for example, to a folder of Expert Advisors or indicators). Inside the file, in the comments,
options are left for connecting certain program assembly configurations. By default, if you do not edit
the file, you will bet an Expert Advisor.

---

## Page 620

Part 5. Creating application programs
620
5.1  General principles for executing MQL programs
//+------------------------------------------------------------------+
//| Uncomment the following line to get the service                  |
//| NB: also activate #define _OnStart OnStart                       |
//+------------------------------------------------------------------+
//#property service
  
//+------------------------------------------------------------------+
//| Uncomment the following line to get a library                    |
//+------------------------------------------------------------------+
//#property library
  
//+------------------------------------------------------------------+
//| Uncomment the following line to get a script or                  |
//| service (#property service must be enabled)                      |
//+------------------------------------------------------------------+
//#define _OnStart OnStart
  
//+------------------------------------------------------------------+
//| Uncomment one of the following two lines for the indicator       |
//+------------------------------------------------------------------+
//#define _OnCalculate1 OnCalculate
//#define _OnCalculate2 OnCalculate
  
#include <MQL5Book/AllInOne.mqh>
If we attach the program to the chart, we will get an entry in the log:
EnumToString((ENUM_PROGRAM_TYPE)MQLInfoInteger(MQL_PROGRAM_TYPE))=PROGRAM_EXPERT / ok
OnInit
OnChartEvent
OnTick
OnTick
OnTick
...
Also, most likely, a stream of records will be generated from the OnTick handler if the market is open.
If you duplicate the mq5 file under a different name and, for example, uncomment the directive
#property service, the compiler will generate the service but will return a few warnings.
no OnStart function defined in the script
OnInit function is useless for scripts
OnDeinit function is useless for scripts
The first of them, about the absence of the OnStart function, is actually significant, because when a
service instance is created, no function will be called in it, but only global variables will be initialized.
However, due to this, the journal (Experts tab in the terminal) will still print the PROGRAM_SERVICE
type. But as a rule, in services, as well as in scripts, it is assumed that the OnStart function is present.
The other two warnings arise because our header file contains handlers for all occasions, and the
compiler reminds us that OnInit and OnDeinit are pointless (will not be called by the terminal and will
not even be included in the binary image of the program). Of course, in real programs there should be
no such warnings, that is, all handlers should be involved, and everything superfluous should be removed

---

## Page 621

Part 5. Creating application programs
621 
5.1  General principles for executing MQL programs
from the source code, either physically or logically, using preprocessor directives for conditional
compilation.
If you create another copy of AllInOne.mq5 and activate not only the #property service directive but
also the #define _ OnStart OnStart macro, you will get a fully working service as a result of its
compilation. When launched, it will not only display the name of its type but also the name of the
triggered handler OnStart.
The macro was required to be able to enable/disable the standard handler OnStart if they wish to. In
the AllInOne.mqh text, this function is described as follows:
void _OnStart() // "extra" underline makes the function customized 
{
   Print(__FUNCTION__);
}
The name starting with an underscore makes it not a standard handler, but just a user-defined function
with a similar prototype. When we include a macro, during compilation the compiler replaces _ OnStart
on OnStart, and the result is already a standard handler. If we explicitly named the OnStart function,
then, according to the priorities of the characteristics that determine the type of the MQL program
(see section Features of MQL programs of various types), it would not allow you to get an Expert
Advisor template (because OnStart identifies the program as a script or service).
Similar custom compilation with macros _ OnCalculate1  or _ OnCalculate2 required to optionally "hide"
the handler with a standard name OnCalculate: otherwise, if it was present, we would always get an
indicator.
f in the next copy of the program you activate the macro #define _ OnCalculate1  OnCalculate, you will
get an example indicator (even though it is empty and does nothing). As we will see later, there are two
different forms of the handler OnCalculate for indicators, in connection with which they are presented
under numbered names (_ OnCalculate1  and _ OnCalculate2). If you run the indicator on the chart, you
can see in the log the names of events OnCalculate (upon arrival of ticks) and OnChartEvent (for
example, on a mouse click).
When compiling the indicator, the compiler will generate two warnings:
no indicator window property is defined, indicator_chart_window is applied
no indicator plot defined for indicator
This is because indicators, as data visualization tools, require some specific settings in their code that
are not here. At this stage of superficial acquaintance with different types of programs, this is not
important. But further on, we will learn how to describe their properties and arrays in indicators, which
determine what and how should be visualized on the chart. Then these warnings will disappear.
Event queue
When a new event occurs, it must be delivered to all MQL programs running on the corresponding
chart. Due to the single-threaded execution model of MQL programs (see section Threads), it may
happen that the next event arrives when the previous one is still being processed. For such cases, the
terminal maintains an event queue for each interactive MQL program. All events in it are processed one
after another in order of receipt.
Event queues have a limited size. Therefore, an irrationally written program can provoke an overflow of
its queue due to slow actions. On overflow, new events are discarded without being queued.

---

## Page 622

Part 5. Creating application programs
622
5.1  General principles for executing MQL programs
Not processing events fast enough can negatively affect the user experience or data quality (imagine
you record Market Depth changes and skip a few messages). To solve this problem, you can look for
more efficient algorithms or use the parallel operation of several interconnected MQL programs (for
example, assign calculations to an indicator, and only read ready-made data in an Expert Advisor).
It should be borne in mind that the terminal does not place all events in the queue but operates
selectively. Some types of events are processed according to the principle "no more than one event of
this type in the queue". For example, if there is already the OnTick event in the queue, or it is being
processed, then a new OnTick event is not queued. If there is already the OnTimer event or a chart
change event in the queue, then new events of these types are also discarded (ignored). It is about a
specific instance of the program. Other, less "busy" programs will receive this message.
We do not provide a complete list of such event types because this optimization by skipping
"overlapping" events can be changed by the terminal developers.
The approach to organizing the work of programs in response to incoming events is called event-driven.
It can also be called asynchronous because the queuing of an event in the program queue and its
extraction (together with processing) occur at different moments (ideally, separated by a microscopic
interval, but the ideal is not always achievable). However, of the four types of MQL programs, only
indicators and Expert Advisors fully follow this approach. Scripts and services have, in fact, only the
main function, which, when called, must either quickly perform the required action and complete or
start an endless loop to maintain some activity (for example, reading data from the network) until the
user stops. We have seen examples of such loops:
while(!IsStopped())
{
  useful code
   ...
   Sleep(...);
} 
In such loops, it is important not to forget to use Sleep with some period to share CPU resources with
other programs. The value of the period is selected based on the estimated intensity of the activity
being implemented.
This approach can be referred to as cyclic or synchronous, or even as real-time, since you can select
the sleep period to provide a constant frequency of data handling, for example:
int rhythm = 100; // 100 ms, 10 times per sec
while(!IsStopped())
{
   const int start = (int)GetTickCount();
  useful code
   ...
   Sleep(rhythm - ((int)GetTickCount() - start));
} 
Of course, the "useful code" must fit in the allotted frame.
In contrast, with the event approach, it is not known in advance when the next time the piece of code
(handler) will work. For example, in a fast market, during the news, ticks can come in batches, and at
night they can be absent for whole seconds. In the limiting case, after the final tick on Friday evening,
the next price change for some financial instrument can be broadcast only on Monday morning, and

---

## Page 623

Part 5. Creating application programs
623
5.1  General principles for executing MQL programs
therefore the events OnTick will be absent for two days. In other words, in events (and moments of
activation of event handlers) there is no regularity, no clear schedule.
But if necessary, you can combine both trips. In particular, the timer event (OnTimer) provides
regularity, and the developer can periodically generate custom events for a chart inside a loop (for
example, flashing a warning label).
5.1 .4 Features of starting and stopping programs of various types
In programming, the term initialization is used in many different contexts. In MQL5, there is also some
ambiguity. In the Initialization section, we have already used this word to mean setting the initial values
of variables. Then we discussed the initialization event OnInit in indicators and Expert Advisors. Although
the meaning of both initializations is similar (bring the program to a working state), they actually mean
different stages of preparing an MQL program for launch: system and application.
The life cycle of a finished MQL program can be represented by the following major steps:
1 .Loading – reading a program from a file into the terminal's memory: this includes instructions,
predefined data (literals), resources, and libraries. This is where #property directives come into
play.
2. Allocating memory for global variables and setting their initial values – it is system initialization
performed by the runtime. Recall that in the section Initialization, while studying the start of the
program under the debugger step by step, we saw that the @global_initializations entry was on
the stack. This was the code block for this item, which was created implicitly by the compiler. If
the program uses global objects of classes/structures, their constructors will be called at this
stage.
3. Calling the OnInit event handler (if it exists): it carries out a higher-level, applied initialization, and
thus each program performs it independently, as necessary. For example, it can be dynamic
memory allocation for arrays of objects, for which, for one reason or another, you need to use
parametric constructors instead of default constructors. As we know, automatic memory
allocation for arrays uses only default constructors, and therefore they cannot be initialized within
the previous step (2). It can also be opening files, calling built-in API functions to enable the
necessary chart modes, etc.
4. A loop until the user closes the program or terminal or performs any other action that requires
reinitialization (see further):
· calling other handlers as appropriate events occur.
5. Calling the OnDeinit event handler (if it exists) upon detection of an attempt to close the program
by the user or programmatically (the corresponding function ExpertRemove is available only in
Expert Advisors and scripts).
6. Finalization: freeing allocated memory and other resources that the programmer did not consider
as necessary to free in OnDeinit. If the program uses OOP, the destructors of global and static
objects are called here.
7.Downloading the program.
Scripts and services a priori do not have OnInit and OnDeinit handlers, and therefore steps 3 and 5 are
absent for them, and step 4 degenerates into a single OnStart call.
System initialization (step 2) is inseparable from loading, that is, it always follows it. Finalization always
precedes unloading. However, indicators and Expert Advisors go through the stages of loading and
unloading differently in different situations. Therefore, OnInit and OnDeinit calls (steps 3 and 5) are the

---

## Page 624

Part 5. Creating application programs
624
5.1  General principles for executing MQL programs
reference points at which it is possible to provide consistent applied initialization and deinitialization of
Expert Advisors and indicators.
Loading of indicators and Expert Advisors is performed in the following cases:
Case
The user launches the program on the chart
+
+
Launching the terminal (if the program was running on the chart before the previous
closing of the terminal)
+
+
Loading a template (if the template contains a program attached to the chart)
+
+
Profile change (if the program is attached to one of the profile charts)
+
+
After successful recompilation, if the program was attached to the chart
+
+
Changing the active account
+
+
-
-
-
Change the symbol or period of the chart to which the indicator is attached
+
-
Changing the input parameters of the indicator
+
-
-
-
-
Connecting to the account (authorization), even if the account number has not
changed
-
+
In a more compact form, the following rule can be formulated: Expert Advisors do not go through the
full life cycle, that is, they do not reload when the symbol/timeframe of the chart changes, as well as
when the input parameters change.
Therefore, a similar asymmetry can be observed when unloading programs. The reasons for unloading
indicators and Expert Advisors are:

---

## Page 625

Part 5. Creating application programs
625
5.1  General principles for executing MQL programs
Case
Removing the program from the chart
+
+
Closing the terminal (when the program is attached to the chart)
+
+
Loading a template on the chart on which the program is running
+
+
Closing the chart on which the program is running
+
+
Changing the profile if the program is attached to one of the charts of the profile
+
+
Changing the account to which the terminal is connected
+
+
-
-
-
Changing the symbol and/or period of the chart to which the indicator is attached
+
-
Changing the input parameters of the indicator
+
-
-
-
-
Attaching another or the same EA to the chart where the current EA is already running
-
+
Calling the ExpertRemove function
-
+
The reason for deinitialization can be found in the program using the function UninitializeReason or flag
_ UninitReason (cm. section Checking the status and reason for stopping an MQL program).
Please note that when you change the symbol or timeframe of the chart, as well as when you change
the input parameters, the Expert Advisor remains in memory, that is, steps 6-7 (finalization and
unloading) and steps 1 -2 (loading and primary memory allocation) are not executed, therefore values of
global and static variables are not reset. In this case, the OnDeinit and OnInit handlers are called
sequentially on the old and on the new symbol/timeframe respectively (or at the old and new settings).
A consequence of global variables not being cleared in Expert Advisors is that the deinitialization code
_ UninitReason remains unchanged for analysis in the OnInit handler. The new code will be written to the
variable only in case of the next event, just before the OnDeinit call.
All events received for the Expert Advisor before the end of the OnInit function, are skipped.
When the MQL program is launched for the first time, the settings dialog is displayed between steps
1  and 2. When changing the input parameters, the settings dialog is wedged into the general loop in
different ways depending on the type of program: for indicators, it still appears before step 2, and
for Expert Advisors — before step 3.
The book is accompanied by an indicator and Expert Advisor template entitled LifeCycle.mq5. It logs
global initialization/finalization steps in OnInit/OnDeinit handlers. Place programs on the chart and
see what events occur in response to various user actions: loading/unloading, changing parameters,
switching symbols/timeframes.

---

## Page 626

Part 5. Creating application programs
626
5.1  General principles for executing MQL programs
The script is loaded only when it is added to the chart. If a script is running in a loop, recompiling it
does not result in a restart.
The service is loaded and unloaded using the context menu commands in the terminal interface. When
a service that is already running is recompiled, it is restarted. Recall that active instances of services
are automatically loaded when the terminal starts and unloaded when closes.
In the next two sections, we will consider the features of launching different MQL programs at the level
of event handlers.
5.1 .5 Reference events of indicators and Expert Advisors: OnInit and OnDeinit
In interactive MQL programs – indicators and Expert Advisors – the environment generates two events
to prepare for launch (OnInit) and stop (OnDeinit). There are no such events in scripts and services
because they do not accept asynchronous events: after control is passed to their single event handler
OnStart and until the end of the work, the execution context of the script/service thread is in the code
of the MQL program. In contrast, for indicators and Expert Advisors, the normal course of work
assumes that the environment will repeatedly call their specific event handling functions (we will discuss
them in the sections on indicators and Expert Advisors), and each time, having taken the necessary
actions, the programs will return control to the terminal for idle waiting for new events.
int OnInit()
Function OnInit is a handler of the event of the same name, which is generated after loading an Expert
Advisor or an indicator. The function can only be defined as needed.
The function must return one of the ENUM_INIT_RETCODE enum values.
Identifier
Description
INIT_SUCCEEDED
Successful initialization, program execution can be continued;
corresponds to value 0
INIT_FAILED
Unsuccessful initialization, execution cannot be continued due to
fatal errors (for example, it was not possible to create a file or an
auxiliary indicator); value 1 
IN IT_P AR AM E TE R S _IN CO R R E CT
Incorrect set of input parameters, program execution is impossible
IN IT_AG E N T_N O T_S U ITAB L E 
Specific code to work in tester: for some reason, this agent is not
suitable for testing (for example, not enough RAM, no OpenCL
support, etc.)
If OnInit returns any non-zero return code, this means unsuccessful initialization, and then the Deinit
event is generated, with deinitialization reason code REASON_INITFAILED (see below).
The OnInit function can be declared with a result type void: in this case, initialization is always
considered successful.
In the OnInit handler, it is important to check that all necessary environment information is
present, and if it is not available, defer preparatory actions for the next tick or timer arrival events.
The point is that when the terminal starts, the OnInit event often triggers before a connection to
the server is established, and therefore many properties of financial instruments and a trading

---

## Page 627

Part 5. Creating application programs
627
5.1  General principles for executing MQL programs
account are still unknown. In particular, the value of one pip of a particular symbol may be returned
as zero.
void OnDeinit(const int reason)
The OnDeinit function (if it is defined) is called when the Expert Advisor or indicator is deinitialized. The
function is optional.
The reason parameter contains the deinitialization reason code. Possible values are shown in the
following table.
Constant
Value
Description
REASON_PROGRAM
0
Expert Advisor stopped operation by
ExpertRemove function call
REASON_REMOVE
1
Program removed from the chart
REASON_RECOMPILE
2
Program recompiled
REASON_CHARTCHANGE
3
Chart symbol or period changed
REASON_CHARTCLOSE
4
Chart closed
REASON_PARAMETERS
5
Input parameters changed
REASON_ACCOUNT
6
Another account has been activated, or a
reconnection to the trading server has occurred
due to a change in the account settings
REASON_TEMPLATE
7
Different chart template applied
REASON_INITFAILED
8
OnInit handler returned a non-null value
REASON_CLOSE
9
Terminal closed
The same code can be obtained anywhere in the program using the UninitializeReason function if the
stop flag _ StopFlag is set in the MQL program.
The AllInOne.mqh file has the Finalizer class which allows you to "hook" the deinitialization code in the
destructor through the UninitializeReason call. We must get the same value in the OnDeinit handler.

---

## Page 628

Part 5. Creating application programs
628
5.1  General principles for executing MQL programs
class Finalizer
{
   static const Finalizer f;
public:
   ~Finalizer()
   {
      PRTF(EnumToString((ENUM_DEINIT_REASON)UninitializeReason()));
   }
};
static const Finalizer Finalizer::f;
For the convenience of translating codes into a string representation (names of reasons) using
EnumToString, enumeration ENUM_DEINIT_REASON with constants from the above table is described
in the Uninit.mqh file. The log will display entries like:
OnDeinit DEINIT_REASON_REMOVE
EnumToString((ENUM_DEINIT_REASON)UninitializeReason())=DEINIT_REASON_REMOVE / ok
When you change the symbol or timeframe of the chart on which the indicator is located, it is
unloaded and loaded again. In this case, the sequence of triggering the event OnDeinit in the old
copy and OnInit is not defined in the new copy. This is due to the specifics of asynchronous event
processing by the terminal. In other words, it may not be entirely logical that a new copy will be
loaded and initialized before the old one is completely unloaded. If the indicator performs some
chart adjustment in OnInit (for example, creates a graphic object), then without taking special
measures, the unloaded copy can immediately "clean up" the chart (delete the object, considering
it to be its own). In the specific case of graphical objects, there is a particular solution: objects can
be given names that include symbol and timeframe prefixes (as well as the checksum of input
variable values), but in the general case it will not work. For a universal solution to the problem,
some kind of synchronization mechanism should be implemented, for example, on global variables or
resources.
When testing indicators in the tester, MetaTrader 5 developers decided not to generate the OnDeinit
event. Their idea is that the indicator can create some graphical objects, which it usually removes in
the OnDeinit handler, but the user would like to see them after the test is completed. In fact, the
author of an MQL program can, if desired, provide similar behavior and leave objects with a positive
check of the mode MQLInfoInteger(MQL_ TESTER). This is strange since the OnDeinit handler is called
after the Expert Advisor test, and the Expert Advisor can delete objects in the same way in OnDeinit.
Now, only for indicators, it turns out that the regular behavior of the OnDeinit handler cannot be
guaranteed in the tester. Moreover, other finalization is not performed, for example, destructors of
global objects are not called.
Thus, if you need to perform a statistics calculation, file saving, or other action after the test run that
was originally intended for the indicator's OnDeinit, you will have to transfer the indicator algorithms to
the Expert Advisor.
5.1 .6 The main function of scripts and services: OnStart
Utility programs – scripts and services – are executed in the terminal by calling their single event
handling function OnStart.

---

## Page 629

Part 5. Creating application programs
629
5.1  General principles for executing MQL programs
void OnStart()
The function has no parameters and does not return any value. It only serves as an entry point to the
application program from the terminal side.
Scripts are intended, as a rule, for one-time actions performed on a chart (later we will study all the
possibilities provided by the chart API). For example, a script can be used to set up a grid of orders or,
conversely, to close all profitable open positions, to automatically apply markup with graphical objects,
or to temporarily hide all objects.
In scripts, you can use constant actions wrapped in an infinite loop, in which, as mentioned earlier, you
should always check the stop sign (_ StopFlag) and periodically release the processor (Sleep). It should
be remembered here that when you turn off and on the terminal, the script will have to be run again.
Therefore, for such constant activity, if it is not directly related to the schedule, it is better to use the
service. The standard technique in the implementation of the service is just an "infinite" loop.
In the previous parts of the book, almost all examples were implemented as scripts. An example of a
service is the program GlobalsWithCondition.mq5 from the section Synchronizing programs using global
variables. We will see another example in the next section about stopping Expert Advisors and scripts
using the ExpertRemove function.
5.1 .7 Programmatic removal of Expert Advisors and scripts: ExpertRemove
If necessary, the developer can organize the stopping and unloading of MQL programs of two types:
Expert Advisors and scripts. This is done using the ExpertRemove function.
void ExpertRemove()
The function has no parameters and does not return a value. It sends a request to the MQL program
execution environment to delete the current program. In fact, this leads to setting the _ StopFlag flag
and stopping the reception (and processing) of all subsequent events. After that, the program is given 3
seconds to properly complete its work: release resources, break loops in algorithms, etc. If the
program does not do this, it will be unloaded forcibly, with the loss of intermediate data.
This function does not work in indicators and services (the program continues to run).
For each function call, the log will contain the entry "ExpertRemove() function called".
The function is primarily used in Expert Advisors that cannot be interrupted in any other way. In the
case of scripts, it is usually easier to break the loop (if there is one) with the break statement. But if
the loops are nested, or the algorithm uses many function calls from one another, it is easier to take
into account the stop flag at different levels in the conditions for continuing calculations, and in case of
an erroneous situation, set this flag using ExpertRemove. If you do not use this built-in flag, in any case,
you would have to introduce a global variable of the same purpose.
The script ScriptRemove.mq5 provides the ExpertRemove usage example.
A potential problem in the operation of the algorithm, which leads to the need to unload the script, is
emulated by the ProblemSource class. ExpertRemove is randomly called in its constructor.

---

## Page 630

Part 5. Creating application programs
630
5.1  General principles for executing MQL programs
class ProblemSource
{
public:
   ProblemSource()
   {
      // simulating a problem during object creation, for example,
      // with the capture of some resources, such as a file, etc.
      if(rand() > 20000)
      {
         ExpertRemove(); // will set _StopFlag to true
      }
   }
};
Further along, objects of this class are created at the global level and inside the helper function.
ProblemSource global; // object may throw an error
   
void SubFunction()
{
   ProblemSource local; //object may throw an error
   // simulate some work (we need to check the integrity of the object!)
   Sleep(1000);
}
Now we use SubFunction in the OnStart operation, inside the loop with the IsStopped condition.
void OnStart()
{
   int count = 0;
   // loop until stopped by the user or the program itself
   while(!IsStopped())
   {
      SubFunction();
      Print(++count);
   }
}
Here is a log example (each run will be different due to randomness):
1
2
3
ExpertRemove() function called
4
Note that if an error occurs while creating the global object, the loop will never execute.
Because Exert Advisors can run in the tester, the ExpertRemove function can also be used in the tester.
Its effect depends on the place of the function call. If this is done inside the OnInit handler, the function
will cancel testing, that is, one run of the tester on the current set of the Expert Advisor parameters.
Such termination is treated as an initialization error. When ExpertRemove is called in any other place of
the algorithm, the Expert Advisor testing will be interrupted early, but will be processed in a regular
way, with OnDeinit and OnTester calls. In this case, the accumulated trading statistics and the value of

---

## Page 631

Part 5. Creating application programs
631 
5.1  General principles for executing MQL programs
the optimization criterion will be obtained, taking into account that the emulated server time
TimeCurrent does not reach the end date in the tester settings.
5.2 Scripts and services
In this chapter, we will summarize and present the full technical information about the scripts and
services that we have already started to get acquainted with in the previous parts of the book.
Scripts and services have the same principles for organizing and executing program code. As we know,
their main function OnStart is also the only one. Scripts and services cannot process other events.
However, there are a couple of significant differences. Scripts are executed in the context of a chart
and have direct access to its properties through built-in variables such as _ Symbol, _ Period, _ Point, and
others. We will study them in the section Chart properties. Services, on the other hand, work on their
own, not tied to any windows, although they have the ability to analyze all charts using special
functions (the same Chart functions can be used in other types of programs: scripts, indicators, and
Expert Advisors).
On the other hand, the created instances of the service are automatically restored by the terminal in
the next sessions. In other words, the service, once started, always remains running until the user
stops it. In contrast, the script is deleted when the terminal is turned off or the chart is closed.
Please note that the service is executed in the terminal, like all other types of MQL programs, and
therefore closing the terminal also stops the service. The active service will resume the next time
you start the terminal. Uninterrupted operation of MQL programs can only be ensured by a
constantly running terminal, for example, on a VPS.
In scripts and services, you can set General properties of programs using #property directives. In
addition to them, there are properties that are specific to scripts and services; we will discuss them in
the next two sections.
The scripts that are currently running on the charts are listed in the same list that shows running
Expert Advisors – in the Experts dialog opened with the Expert List command of the chart context menu.
From there, they can be forcibly removed from the chart.
Services can only be managed from the Navigator window.
5.2.1  Scripts
A script is an MQL program with the only handler OnStart, provided there is no #property
servicedirective  (otherwise you get a service, see the next section).
By default, the script immediately starts executing when it is placed on the chart. The developer can
ask the user to confirm the start by adding the #property script_ show_ confirm directive to the
beginning of the file. In this case, the terminal will show a message with the question "Are you sure you
want to run 'program' on chart 'symbol, timeframe'?" and buttons Yes and No.
Scripts, like other programs, can have input variables. However, for scripts, the parameter input dialog
is not shown by default, even if the script defines inputs. To ensure that the properties dialog opens
before running the script, the #property script_ show_ inputs directive should be applied. It takes
precedence over script_ show_ confirm, that is, the output of the dialog disables the confirmation
request (since the dialog itself acts in a similar role). The directive calls a dialog even if there are no

---

## Page 632

Part 5. Creating application programs
632
5.2 Scripts and services
input variables. It can be used to show the product description and version (they are displayed on the
Common tab) to the user.
The following table shows combination options for the #property directive and their effect on the
program.
Directive
Effect
script_show_confirm
script_show_inputs
Immediate launch
No
No
Confirmation request
Yes
No
Opening the properties dialog
irrelevant
Yes
A simple example of a script with directives is in the file ScriptNoComment.mq5. The purpose of the
script is as follows. Sometimes MQL programs leave behind unnecessary comments in the upper left
corner of the chart. Comments are stored in chr-files along with the chart, so even after restarting the
terminal they are restored. This script allows you to clear a comment or set it to an arbitrary value. If
you Assign hotkey to a script using the Navigator context menu command, it will be possible to clean
the comment of the current chart with one click.
Originally, directives script_ show_ confirm and script_ show_ inputs are disabled by becoming inline
comments. You can experiment with different combinations of directives by uncommenting them one at
a time or at the same time.
//#property script_show_confirm
//#property script_show_inputs
   
input string Text = "";
   
void OnStart()
{
   Comment(""); // clean up the comment
}
5.2.2 Services
A service is an MQL program with a single OnStart handler and the #property service directive.
Recall that after the successful compilation of the service, you need to create and configure its
instance (one or more) using the Add Service command in the context menu of the Navigator window.
As an example of a service, let's solve a small applied problem that often arises among developers of
MQL programs. Many of them practice linking their programs to the user's account number. This is not
necessarily about a paid product but may refer to distribution among friends and acquaintances to
collect statistics or successful settings. At the same time, the user can register demo accounts in
addition to a working real account. The lifetime of such accounts is usually limited, and therefore it is
rather inconvenient to update the link for them every couple of weeks. To do this, you need to edit the
source code, compile and send the program again.
Instead, we can develop a service that will register in global variables (or files) the numbers of accounts
to which a successful connection was implemented from the given terminal.

---

## Page 633

Part 5. Creating application programs
633
5.2 Scripts and services
The binding technology is based on pairwise encryption (or, alternatively, hashing) of account numbers:
the old login account and the new login account. The previous account must be a master account (to
which the conditional link is "issued") in order for the pair's common signature to extend the rights to
use the product to the new account. The key is a secret known only inside the programs (it is assumed
that all of them are supplied in a closed, compiled form). The result of the operation will be a string in
the Base64 format. The implementation uses MQL5 API functions, some of which are yet to be studied,
in particular, obtaining an account number via AccountInfoInteger and CryptEncode encryption function.
Connection to the server is checked using the TerminalInfoInteger function (see Checking network
connections).
The service is not required to know which accounts are master, and which ones are additional ones. It
only needs to “sign” pairs of any successively logged-in accounts in a special way. But a specific
application program should supplement the process of checking its "license": in addition to comparing
the current account with the master account, you should repeat the service algorithm: create a pair
[master account; current account], calculate the encrypted signature for it, and check whether it is
among the global variables.
It will be possible to steal such a license by transferring it to another computer only if you connect to
the same account in trading mode (not investor). An unscrupulous user, of course, can create demo
accounts for other people. Therefore, it is desirable to improve the protection. In the current
implementation, the global variable is simply made temporary, that is, it is deleted along with the end of
the terminal session, but this does not prevent its possible copying.
As additional measures, it is possible, for example, to encrypt the time of its creation in the signature
and provide for the expiration of rights every day (or with another frequency). Another option is to
generate a random number when the service starts and add it to the signed information along with
account numbers. This number is known only inside the service, but it can translate it to interested
MQL programs on charts using the EventChartCustom function. Thus, the signature will continue to be
valid in this instance of the terminal until the end of the session. Each session will generate and send a
new random number, so it will not work for other terminals. Finally, the simplest and most convenient
option would probably be to add to the signature of the system start time: (TimeLocal() -
GetTickCount() / 1 000) or its derivative.
Of the various types of MQL programs, only some continue to run between account switches and allow
this protection scheme to be implemented. Since it is necessary to protect MQL programs of any type
in a uniform way, including indicators and Expert Advisors (which are reloaded when the account is
changed), it makes sense to entrust this task to a service. Then the service, which is constantly
running from the moment the terminal is loaded until it is closed, will control logins and generate
authorizing signatures.
The source code of the service is given in the file MQL5/Services/MQL5Book/p5/ServiceAccount.mq5.
The input parameters specify the master account and the prefix of global variables in which signatures
will be stored. In real programs, lists of master accounts should be hardcoded in the source code, and
instead of global variables, it is better to use files in the Common folder to cover the tester as well.
#property service
   
input long MasterAccount = 123456789;
input string Prefix = "!A_";
The main function of the service performs its work as follows: in an endless loop with pauses of 1 
second, we track account changes and save the last number, create a signature for the pair, and write
it to a global variable. The signature is created by the Cipher function.

---

## Page 634

Part 5. Creating application programs
634
5.2 Scripts and services
void OnStart()
{
   static long account = 0; // previous login
   
   for(; !IsStopped(); )
   {
      // require connection, successful login and full access (not investor)
      const bool c = TerminalInfoInteger(TERMINAL_CONNECTED)
                  && AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
      const long a = c ? AccountInfoInteger(ACCOUNT_LOGIN) : 0;
   
      if(account != a) // account changed
      {
         if(a != 0) // current account
         {
            if(account != 0) // previous account
            {
               // transfer authorization from one to another
               const string signature = Cipher(account, a);
               PrintFormat("Account %I64d registered by %I64d: %s", 
                  a, account, signature);
               // saving a record about the connection of accounts
               if(StringLen(signature) > 0)
               {
                  GlobalVariableTemp(Prefix + signature);
                  GlobalVariableSet(Prefix + signature, account);
               }
            }
            else // the first account is authorized, now waiting for the second one
            {
               PrintFormat("New account %I64d detected", a);
            }
            // remember the last active account
            account = a;
         }
      }
      Sleep(1000);
   }
}
The Cipher function uses a special union ByteOverlay2 to represent a pair of account numbers (of type
long) as a byte array, which is passed for encryption in CryptEncode (CRYPT_DES encryption method is
chosen here, but it can be replaced with CRYPT_AES1 28, CRYPT_AES256 or just
CRYPT_HASH_SHA256 hashing (with secret as "salt"), if information recovery from "signature" is not
required).