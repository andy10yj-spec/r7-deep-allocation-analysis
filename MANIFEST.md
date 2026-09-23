# r7 deep-allocation analysis bundle

| kind | path | role | md5 | bytes |
|---|---|---|---|---|
| production | code/production/tk_r7.py | production 탐색기 (이번 분석 대상, 수정 안 함) | 01fcff1e7287e44ed1af882230cc1cf1 | 32411 |
| prototype | code/prototype/tk_r7_promo.py | P1 rollout->tree promotion prototype (채택 안 함) | 8ed2ab01a36ef697757dca6704d28ef0 | 33017 |
| diagnostic | code/diagnostic/r7_deep_diag.py | deep-node / expansion 계측 wrapper (deep JSON 생성) | bb868f1a9283946adc86abf529c00841 | 12180 |
| diagnostic | code/diagnostic/r7_trajectory.py | 후보별 value trajectory 계측 (trajectory JSON 생성) | 9ad2298935a85d624ad2b279210f840c | 10549 |
| diagnostic | code/diagnostic/adaptive_replay.py | trajectory 적재 + nmcs_pass selection 재현 (아래 분석 스크립트가 import) | 22791732e82b69a58ea266397b8385b2 | 20560 |
| diagnostic | code/diagnostic/r7_alloc_offline.py | deep allocation 3축 분리 분석 | bad7b245726b15004e2a2c88cbbb44c8 | 15683 |
| diagnostic | code/diagnostic/r7_callend.py | NMCS call 종료 commit 분석 | cb4e9029451adc8c4714d92fdc99c008 | 5184 |
| diagnostic | code/diagnostic/r7_tie_cf.py | terminal-ending tie 복원 + offline counterfactual (이번 단계) | 89304db2a46e1c4417e24fb3910d0524 | 10476 |
| result | results/alloc.txt | r7_alloc_offline.py 출력 | e9d7f82c35b8a7dc7d358abdf80275b2 | 12509 |
| result | results/callend.txt | r7_callend.py 출력 | c9e3df4e32a4379dac2f8bd28ac58e68 | 2297 |
| result | results/tie_cf.txt | r7_tie_cf.py 출력 | 3ef707ad6b8857c3bfc8b0d69337071e | 5370 |
| result | results/tie_cf.json | 106 terminal-ending commit 레코드 | dc14fe5fd02dc4a305e8e4fb54cce143 | 39973 |
| data-full | data/full/traj_big_20260917.json | 원본 JSON (전체) | f414c2d68bd7baabfe124a5f0065c65d | 3094308 |
| data-full | data/full/traj_big_12345.json | 원본 JSON (전체) | 3d6acf956d5f333ee861f2af615eb1a6 | 3098732 |
| data-full | data/full/traj_big_777.json | 원본 JSON (전체) | fc47500493e9a77babcbba5f9a6c4f9a | 3090036 |
| data-full | data/full/deep_chundra_12345.json | 원본 JSON (전체) | 2090a576f5265da062691e0690837d64 | 6469282 |
| data-full | data/full/deep_chundra_20260917.json | 원본 JSON (전체) | 0b2519c65b42efa67dd43887cbe56618 | 6829672 |
| data-full | data/full/deep_chundra_777.json | 원본 JSON (전체) | 43678bfe718725634689aea0df9c479e | 6441891 |
| data-full | data/full/deep_sangen_12345.json | 원본 JSON (전체) | 2f661419753c7a6b2e9221f8825ed517 | 4544296 |
| data-full | data/full/deep_sangen_20260917.json | 원본 JSON (전체) | 555d08d86fbbcd458ace01db80f1169f | 5513880 |
| data-full | data/full/deep_sangen_777.json | 원본 JSON (전체) | fd37cd701c3ae58b08ba576da15f809b | 4326838 |
| data-full | data/full/deep_sangen_8888.json | 원본 JSON (전체) | a9862bdcac5ca62332d94ed454c57f4a | 3204244 |
| data-subset | data/subset/traj_big_20260917.subset.json | 분석에 쓰인 필드만 남긴 subset | 3c96c73dd23b1ee6f3c2936244e043e2 | 624821 |
| data-subset | data/subset/traj_big_12345.subset.json | 분석에 쓰인 필드만 남긴 subset | ca76879a2d3db50c0dbec84956c7a730 | 639044 |
| data-subset | data/subset/traj_big_777.subset.json | 분석에 쓰인 필드만 남긴 subset | 2e259b05ca28a0c6809d2b3c85ad54f3 | 643804 |
| data-subset | data/subset/deep_chundra_12345.subset.json | 분석에 쓰인 필드만 남긴 subset | e351ea43b59aabf1283d2f5deebf5cd8 | 1250319 |
| data-subset | data/subset/deep_chundra_20260917.subset.json | 분석에 쓰인 필드만 남긴 subset | 473e80103a7602bb8518051503f608cd | 1322628 |
| data-subset | data/subset/deep_chundra_777.subset.json | 분석에 쓰인 필드만 남긴 subset | 340206860ff5e4997f1c8eff14b2d397 | 1243150 |
| data-subset | data/subset/deep_sangen_12345.subset.json | 분석에 쓰인 필드만 남긴 subset | 66bdbb7d03c06ad4161f5b8e7d6b90b3 | 846252 |
| data-subset | data/subset/deep_sangen_20260917.subset.json | 분석에 쓰인 필드만 남긴 subset | b4fd3bc10dbe637316fa1ec61327e4f5 | 1040413 |
| data-subset | data/subset/deep_sangen_777.subset.json | 분석에 쓰인 필드만 남긴 subset | 90b26a75b1bd5c389137eeddfcf8ebf8 | 798450 |
| data-subset | data/subset/deep_sangen_8888.subset.json | 분석에 쓰인 필드만 남긴 subset | 08c7f0149855158c4bd4f664becb8e92 | 673758 |
