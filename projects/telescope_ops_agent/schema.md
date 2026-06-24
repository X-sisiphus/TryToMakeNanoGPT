# TelescopeOps-Agent 数据 Schema v0.1

## Case Schema

每个样本是一个诊断 case。

```json
{
  "case_id": "case_0001",
  "question": "昨晚 02:10 到 02:40 观测质量下降，请帮我定位原因。",
  "context": {
    "telescope": "TJU-1m",
    "date": "2026-06-25",
    "target": "M31",
    "observation_mode": "imaging"
  },
  "logs": [
    {
      "log_id": "log_0001",
      "source": "observation_log",
      "time": "02:12",
      "text": "Image FWHM increased from 2.1 arcsec to 4.8 arcsec."
    }
  ],
  "gold": {
    "fault_type": "seeing_degradation",
    "affected_subsystem": ["weather"],
    "time_window": "02:10-02:40",
    "evidence_ids": ["log_0001"],
    "diagnosis": "Seeing degradation likely caused image quality drop.",
    "recommended_actions": [
      "Check weather and seeing monitor records.",
      "Compare images before and after 02:10."
    ]
  }
}
```

## Log Source 枚举

```text
observation_log
weather_log
device_log
ccd_log
dome_log
mount_log
focus_log
power_log
network_log
schedule_log
maintenance_note
manual
historical_case
```

## Fault Type 枚举

第一版只保留 8 类：

```text
weather_humidity
seeing_degradation
ccd_temperature
dome_tracking
pointing_error
focus_drift
power_or_network
schedule_conflict
```

## Prediction Schema

模型或系统输出统一转成这个格式评估：

```json
{
  "case_id": "case_0001",
  "fault_type": "seeing_degradation",
  "affected_subsystem": ["weather"],
  "time_window": "02:10-02:40",
  "evidence_ids": ["log_0001", "log_0003"],
  "diagnosis": "The image degradation is most likely caused by poor seeing.",
  "recommended_actions": [
    "Check seeing monitor.",
    "Inspect nearby weather records."
  ],
  "uncertainty": "No direct seeing sensor record is available."
}
```

## 评估字段

第一版评估：

```text
fault_type
affected_subsystem
evidence_ids
```

暂不严格评估：

```text
diagnosis 自然语言质量
recommended_actions 完整性
uncertainty 校准
```

这些留到 agent 和 OPD 阶段再评估。
