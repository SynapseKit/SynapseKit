def test_smoke_sort(benchmark):
    data = list(range(1000, 0, -1))
    benchmark(sorted, data)
