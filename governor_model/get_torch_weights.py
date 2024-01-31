from PerformancePredictor import build_perf_predictors

s1, s2, s3 = build_perf_predictors()

for param in s1.parameters():
    print(param)
for param in s2.parameters():
    print(param)
for param in s3.parameters():
    print(param)
