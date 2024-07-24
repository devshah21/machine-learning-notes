# Metrics

## classification metrics

### accuracy

- accuracy is the most obvious one to compute — it’s essentially the ratio of correctly predicted instances to the total number of instances
    - the formula is: accuracy = correct predictions / total predictions
- this is particularly useful when the class distribution is balanced and all errors are equally important

### precision

- precision is also known as positive predictive value
    - it measures the proportion of correctly predicted positive instances among the total predicted positive instances
        - precision = true positives / (true positives + false positives)
- precision is important when the cost of false positives are high
    - for example, spam detection — a false positive would mean a legitimate email was marked as a spam

### f1 score

- f1 score is an harmonic mean of precision and recall — it provides a single metric that balances both
    - f1 score = 2 * (precision * recall) / (precision + recall)
- f1-score is useful when you need a balance between precision & recall, especially when the classes are imbalanced