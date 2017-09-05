# Machine Learning JS: Challenge #1
The goal is to implement a Perceptron in JS that classifies [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist).

## Files
`features.csv`: Each row is a grayscale image the digit with 8x8 pixels.

`labels.csv`: Contains the labels for the rows in `features.csv`, from 0 to 9, corresponding to the digit.

## Instructions
Run `npm install` to install dependencies.

Open the file `exercise.js` and fill the code snippets indicated by comments, starting from **STEP 1**.

After that, you can test the code by running `node exercise.js` and observing the output in the console.

**Pro Tip:** Pay attention to the Delta parameter in the console messages. It should decrease with each iteration, indicating that the learning is converging.

If you get this right, you should see something like that printed to the console: `Score on test data: 99.27%`.

## Troubleshooting
### My score is 0% / My delta is NaN
The gradient calculation isn't returning a valid number.

### My Delta isn't decreasing
It is possible that the Delta will increase in some iterations, by it should not happen on most of them. If you notice that Delta is increasing most of the time, your gradient calculation might be wrong.

### I can't do this, no matter what :(
It's alright buddy. You can peek the solution in the `answer.js` file.
