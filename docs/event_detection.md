Event Detection Module

## Goal
Detect important match events using tracking data:
- Player positions
- Ball position
- Player IDs
- Timestamps

---

# Ball Events

## 1. Pass Detection

### Input:
- Ball position
- Player positions
- Player IDs

### Logic:
- Ball leaves player A
- Ball moves toward player B
- Player B gains possession

### Output:

```json
{
 "event": "pass",
 "player_from": 12,
 "player_to": 7,
 "timestamp": 32.5
}
2. Shot Detection
Input:
Ball speed
Ball direction
Goal position
Logic:
Ball velocity increases
Ball trajectory moves toward goal
Output:
{
 "event": "shot",
 "player_id": 12,
 "target": "goal",
 "timestamp": 45.2
}
3. First Touch Detection
Input:
Player position
Ball position
Logic:
Ball approaches player
Player gains control
Output:
{
 "event": "first_touch",
 "player_id": 12
}
4. Turnover Detection
Logic:
Current player loses possession
Opponent gains possession
Output:
{
 "event": "turnover",
 "lost_by": 12,
 "won_by": 8
}
Tactical Events
Formation Detection
Detect team formations:
4-3-3
4-2-3-1
3-5-2
Input:
Player pitch coordinates
Output:
{
 "formation": "4-3-3"
}
Press Detection
Logic:
Multiple defenders approach ball carrier
Distance below threshold
Output:
{
 "event": "press",
 "players": [1,5,8]
}
Team Metrics
Calculate:
Possession
Pass Count
Shot Count
Attack Count
Compactness
Defensive Coverage
Database Events Table
Field	Description
id	Event ID
match_id	Match ID
timestamp	Event time
event_type	Pass / Shot / Press
player_id	Player ID
extra_data	JSON details

Pipeline
Video
↓
Detection
↓
Tracking
↓
Event Detection
↓
Database
↓
Dashboard
```