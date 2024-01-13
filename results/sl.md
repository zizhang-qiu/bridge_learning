| activation | hidden_size | num_hidden_layer | dropout | layer_norm | optimizer | lr   | batch_size | acc   |
|------------|-------------|------------------|---------|------------|-----------|------|------------|-------|
| gelu       | 1024        | 4                | false   | false      | adam      | 2e-4 | 128        | 93.93 |
| gelu       | 2048        | 4                | false   | false      | adam      | 3e-4 | 1024       | 94.46 |
| gelu       | 2048        | 6                | false   | false      | adam      | 3e-4 | 1024       | 94.73 |
| gelu       | 2048        | 4                | false   | true       | adam      | 3e-4 | 1024       | 94.72 |
| gelu       | 2048        | 6                | false   | true       | adam      | 3e-4 | 1024       | 95.16 |
| gelu       | 2048        | 6                | 0.5     | true       | adam      | 3e-4 | 1024       | 95.55 |
|            |             |                  |         |            |           |      |            |       |
|            |             |                  |         |            |           |      |            |       |
