"""Scene State Engine — 将提取的证据列表转为结构化的场景状态 JSON。

系统必须"先建立案件世界，再生成图片"。所有空间关系映射表写死在代码中，
不做 LLM 调用，确保确定性和可审计性。

输入：EvidenceItem 列表（来自 evidence_filter.py）
输出：SceneState JSON，可直接用于图像生成 prompt 构建
"""

from datetime import datetime, timezone
from pydantic import BaseModel

from app.services.evidence_filter import EvidenceItem


# ==============================================================================
# 空间关系映射表（写死在代码中）
# ==============================================================================

LOCATION_MAP: dict[str, dict[str, str]] = {
    # 床相关
    "under_bed": {"area": "lower_center", "zone": "bed_area"},
    "under_bed_left": {"area": "lower_left", "zone": "bed_area"},
    "under_bed_right": {"area": "lower_right", "zone": "bed_area"},
    "on_bed": {"area": "center", "zone": "bed_area"},
    # 桌子相关
    "on_desk": {"area": "upper_right", "zone": "desk_area"},
    "under_desk": {"area": "lower_right", "zone": "desk_area"},
    # 窗户相关
    "near_window": {"area": "upper_center", "zone": "window_area"},
    "on_windowsill": {"area": "top_center", "zone": "window_area"},
    # 门相关
    "near_door": {"area": "left_center", "zone": "door_area"},
    "behind_door": {"area": "far_left", "zone": "door_area"},
    # 地面
    "on_floor": {"area": "center", "zone": "floor_area"},
    "floor_left": {"area": "left_center", "zone": "floor_area"},
    "floor_right": {"area": "right_center", "zone": "floor_area"},
    # 柜子/抽屉
    "in_drawer": {"area": "right_center", "zone": "storage_area"},
    "on_shelf": {"area": "upper_right", "zone": "storage_area"},
    # 沙发/茶几区
    "on_sofa": {"area": "center_left", "zone": "sofa_area"},
    "under_sofa": {"area": "lower_left", "zone": "sofa_area"},
    "on_coffee_table": {"area": "center", "zone": "sofa_area"},
    "under_coffee_table": {"area": "lower_center", "zone": "sofa_area"},
    # 接待台/前台区
    "on_reception_desk": {"area": "upper_center", "zone": "reception_area"},
    "behind_reception_desk": {"area": "upper_left", "zone": "reception_area"},
    "near_reception": {"area": "center_left", "zone": "reception_area"},
    # 衣物相关
    "on_clothing": {"area": "center", "zone": "clothing_area"},
    "under_clothing": {"area": "lower_center", "zone": "clothing_area"},
    "in_wardrobe": {"area": "right_center", "zone": "storage_area"},
    "on_hanger": {"area": "upper_right", "zone": "clothing_area"},
    # 血迹/痕迹（不是物品，是位置描述）
    "blood_on_floor": {"area": "center", "zone": "floor_area"},
    "blood_on_wall": {"area": "center_right", "zone": "wall_area"},
    "blood_near_door": {"area": "left_center", "zone": "door_area"},
    # 厨房区
    "on_kitchen_counter": {"area": "upper_right", "zone": "kitchen_area"},
    "in_kitchen_sink": {"area": "upper_center", "zone": "kitchen_area"},
    "on_stove": {"area": "upper_left", "zone": "kitchen_area"},
    # 浴室区
    "in_bathroom": {"area": "right_center", "zone": "bathroom_area"},
    "near_toilet": {"area": "lower_right", "zone": "bathroom_area"},
    "in_bathtub": {"area": "center_right", "zone": "bathroom_area"},
    # 通用地面/墙面
    "against_wall": {"area": "top_center", "zone": "wall_area"},
    "corner_left": {"area": "far_left", "zone": "floor_area"},
    "corner_right": {"area": "far_right", "zone": "floor_area"},
}

DEFAULT_POSITION: dict[str, str] = {"area": "center", "zone": "floor_area"}


# ==============================================================================
# 房间类型推断
# ==============================================================================

ROOM_KEYWORDS: dict[str, str] = {
    "卧室": "bedroom",
    "卧房": "bedroom",
    "主卧": "bedroom",
    "客厅": "living_room",
    "起居室": "living_room",
    "厨房": "kitchen",
    "书房": "office",
    "办公室": "office",
    "卫生间": "bathroom",
    "浴室": "bathroom",
    "洗手间": "bathroom",
    "走廊": "hallway",
    "楼道": "hallway",
    "阳台": "balcony",
    "包房": "ktv_room",
    "KTV": "ktv_room",
    "仓库": "warehouse",
    "车库": "garage",
    "地下室": "basement",
    "室外": "outdoor",
    "接待室": "reception_room",
    "会客室": "reception_room",
    "接待区": "reception_room",
    "前台": "reception_room",
    "会议室": "conference_room",
}


# ==============================================================================
# 输出 Pydantic 模型
# ==============================================================================

class SceneObject(BaseModel):
    id: str                         # 如 "knife_1", "phone_2"
    object_type: str                # 物品类型（与 evidence_type 对应）
    description: str                # 中文描述
    location_key: str               # 对应 LOCATION_MAP 的键
    render_position: dict           # 由 LOCATION_MAP 解析出的渲染位置
    state: dict                     # {"bloody": True, "broken": False} 等
    evidence_category: str          # extractable / uncertain


class SceneState(BaseModel):
    scene_id: str
    scene_name: str                 # 如 "卧室"
    base_room_type: str             # bedroom / living_room / office / outdoor
    objects: list[SceneObject]
    created_at: str


# ==============================================================================
# 英文翻译映射（用于图像生成 prompt）
# ==============================================================================

_ROOM_TYPE_EN: dict[str, str] = {
    "bedroom": "A typical bedroom with a bed, nightstands, and wardrobe",
    "living_room": "A typical living room with sofa, coffee table, and TV stand",
    "kitchen": "a kitchen with countertops, appliances, and sink",
    "office": "an office with desks and office equipment",
    "bathroom": "a bathroom with tiles, toilet, and sink",
    "corridor": "A narrow corridor with doors on both sides",
    "hallway": "an interior hallway or corridor",
    "balcony": "an outdoor balcony space",
    "ktv_room": "A KTV private room with sofa, coffee table, and TV screen",
    "warehouse": "a storage warehouse with shelving",
    "garage": "an enclosed garage space",
    "basement": "A basement with concrete walls and dim lighting",
    "outdoor": "An outdoor scene with natural lighting",
    "stairwell": "A stairwell with steps and railings",
    "reception_room": "a reception room / lobby area with a front desk, chairs, and professional setting",
    "conference_room": "a conference room with a meeting table and chairs",
}

_OBJ_TYPE_EN: dict[str, str] = {
    "物证": "physical evidence item",
    "书证": "documentary evidence",
    "现场结构": "structural element",
    "空间关系": "spatial marker",
}

# ⭐ 常见物证类型 → 英文详细描述映射
# 用于增强 scene prompt 中物品的描述准确性
# 格式: (中文关键词, 英文物品类型描述)
# 优先级：长关键词在前
_ITEM_TYPE_DESC: list[tuple[str, str]] = [
    # 凶器/工具类
    ("水果刀", "a small kitchen fruit knife with a metal blade and plastic handle"),
    ("菜刀", "a kitchen chef knife with a stainless steel blade"),
    ("匕首", "a dagger with a sharp double-edged blade"),
    ("尖刀", "a sharp pointed knife"),
    ("水果刀", "a small paring knife"),
    ("美工刀", "a utility knife with a retractable blade"),
    ("剪刀", "a pair of metal scissors"),
    ("斧头", "an axe with a wooden handle"),
    ("锤子", "a hammer with a metal head"),
    ("铁棍", "an iron rod or metal bar"),
    ("木棍", "a wooden stick or club"),
    ("螺丝刀", "a screwdriver"),
    ("扳手", "a wrench"),
    ("砖块", "a brick"),
    ("玻璃碎片", "broken glass shards"),
    ("酒瓶", "a glass wine bottle"),
    ("啤酒瓶", "a glass beer bottle"),

    # 电子设备类
    ("智能手机", "a modern smartphone with a touchscreen"),
    ("苹果手机", "an iPhone smartphone"),
    ("安卓手机", "an Android smartphone"),
    ("手机", "a mobile smartphone with a glass screen"),
    ("破碎的手机", "a broken smartphone with a cracked screen"),
    ("手机屏幕", "a cracked smartphone screen"),
    ("笔记本电脑", "a laptop computer"),
    ("电脑", "a desktop computer with monitor"),
    ("平板电脑", "a tablet computer"),
    ("U盘", "a USB flash drive"),
    ("移动硬盘", "a portable external hard drive"),
    ("相机", "a digital camera"),
    ("录音笔", "a digital voice recorder"),
    ("耳机", "a pair of earphones"),
    ("充电器", "a phone charger with cable"),
    ("数据线", "a USB data cable"),
    ("电话", "a telephone"),
    ("对讲机", "a walkie-talkie radio"),

    # 衣物/纺织品类
    ("血衣", "a bloodstained piece of clothing"),
    ("上衣", "a shirt or top garment"),
    ("外套", "a jacket or coat"),
    ("裤子", "a pair of trousers or pants"),
    ("T恤", "a cotton T-shirt"),
    ("衬衫", "a button-up dress shirt"),
    ("牛仔裤", "a pair of blue jeans"),
    ("内裤", "a pair of underwear"),
    ("内衣", "undergarments"),
    ("袜子", "a pair of socks"),
    ("手套", "a pair of gloves"),
    ("帽子", "a hat or cap"),
    ("围巾", "a scarf"),
    ("毛巾", "a bath towel"),
    ("纸巾", "tissue paper"),
    ("床单", "a bed sheet"),
    ("被套", "a duvet cover"),
    ("枕头", "a pillow"),

    # 鞋类
    ("拖鞋", "a pair of slippers"),
    ("运动鞋", "a pair of athletic sneakers"),
    ("皮鞋", "a pair of leather shoes"),
    ("高跟鞋", "a pair of high-heeled shoes"),
    ("靴子", "a pair of boots"),
    ("凉鞋", "a pair of sandals"),

    # 容器/器皿类
    ("玻璃杯", "a clear drinking glass"),
    ("水杯", "a drinking cup"),
    ("保温杯", "a thermos cup"),
    ("塑料杯", "a plastic cup"),
    ("陶瓷杯", "a ceramic mug"),
    ("酒瓶", "a glass liquor bottle"),
    ("药瓶", "a medicine bottle"),
    ("花瓶", "a glass vase"),
    ("碗", "a bowl"),
    ("盘子", "a plate"),
    ("筷子", "a pair of chopsticks"),
    ("勺子", "a spoon"),

    # 家具类
    ("椅子", "a chair"),
    ("凳子", "a wooden stool"),
    ("桌子", "a table"),
    ("书桌", "a desk"),
    ("沙发", "a sofa"),
    ("茶几", "a coffee table"),
    ("床", "a bed"),
    ("衣柜", "a wardrobe"),
    ("柜子", "a cabinet"),
    ("抽屉", "a drawer"),
    ("书架", "a bookshelf"),

    # 日常用品类
    ("钥匙", "a set of keys"),
    ("钱包", "a wallet"),
    ("手提包", "a handbag"),
    ("背包", "a backpack"),
    ("手表", "a wristwatch"),
    ("眼镜", "a pair of eyeglasses"),
    ("戒指", "a ring"),
    ("项链", "a necklace"),
    ("打火机", "a lighter"),
    ("香烟", "a cigarette"),
    ("烟头", "a cigarette butt"),
    ("火柴", "a box of matches"),
    ("雨伞", "an umbrella"),
    ("手电筒", "a flashlight"),

    # 药品/化学品类
    ("药品", "medication or pills"),
    ("药片", "pills or tablets"),
    ("胶囊", "medicine capsules"),
    ("注射器", "a syringe"),
    ("针管", "a needle and syringe"),
    ("农药", "a bottle of pesticide"),
    ("毒药", "a bottle of toxic substance"),

    # 文件/纸张类
    ("纸条", "a piece of paper note"),
    ("信件", "a letter"),
    ("笔记本", "a notebook"),
    ("日记本", "a diary"),
    ("合同", "a contract document"),
    ("身份证", "an ID card"),
    ("银行卡", "a bank card"),
    ("名片", "a business card"),

    # 食物类
    ("水果", "fresh fruit"),
    ("饭菜", "food and dishes"),
    ("饮料", "a beverage drink"),
    ("水杯", "a cup with liquid"),

    # 结构类
    ("门", "a door"),
    ("窗户", "a window"),
    ("门锁", "a door lock"),
    ("窗户锁", "a window lock"),
    ("墙面", "a wall"),
    ("地板", "a floor"),
    ("天花板", "a ceiling"),
    ("楼梯", "a staircase"),
    ("走廊", "a hallway corridor"),
]


def _enrich_item_description(description: str) -> str:
    """根据中文描述匹配并添加英文物品类型详情，提升生成准确性。

    Args:
        description: 中文物品描述，如 "水果刀"、"破碎的手机"

    Returns:
        str: 增强后的英文描述（如果匹配到），否则返回原文
    """
    # ⭐ 按关键词长度从长到短排序，确保更具体的描述优先匹配
    # 例如："破碎的手机" 优先于 "手机"
    sorted_patterns = sorted(
        _ITEM_TYPE_DESC,
        key=lambda x: len(x[0]),
        reverse=True,
    )
    for cn_kw, en_desc in sorted_patterns:
        if cn_kw in description:
            return en_desc
    return description

_STATE_ADJ_EN: dict[str, str] = {
    "bloody": "bloodstained",
    "broken": "broken",
    "dirty": "soiled",
    "wet": "wet",
    "burnt": "charred",
    "rusty": "rusted",
    "torn": "torn",
    "shattered": "shattered",
    "cracked": "cracked",
    "stained": "stained",
    "folded": "folded",
    "open": "open",
    "closed": "closed",
    "locked": "locked",
    "empty": "empty",
    "full": "full",
}

# ⭐ 中文状态键 → 英文描述（三级：安全版 / 中等版 / 详细版）
# 安全版：极简，确保通过安全过滤
# 中等版：增加细节但避免高风险词，在安全和效果之间平衡
# 详细版：完整细节，用于安全过滤宽松的模型

_CN_STATE_PATTERNS_SAFE: list[tuple[str, str]] = [
    ("喷溅血迹", "with reddish-brown splash marks"),
    ("擦拭状血迹", "with smudged reddish-brown marks"),
    ("滴落状血迹", "with drip marks"),
    ("血迹分布", "with distributed discoloration"),
    ("干涸血迹", "with dark brown dried marks"),
    ("氧化血迹", "with dark brown oxidized marks"),
    ("接触性出血", "with contact transfer marks"),
    ("鲜血", "with red-brown wet marks"),
    ("血迹", "with reddish-brown marks"),
    ("血痕", "with reddish-brown marks"),
    ("带血", "with red-brown marks"),
    ("血", "with reddish-brown marks"),
    ("唾液印记", "with dark moist marks"),
    ("唾液斑", "with dark moist marks"),
    ("唾液", "with dark wet marks"),
    ("精斑", "with biological marks"),
    ("精液", "with biological marks"),
    ("凌乱", "in a disarranged state"),
    ("混乱", "in a disarranged state"),
    ("翻倒", "overturned"),
    ("打斗痕迹", "with signs of disturbance"),
    ("挣扎痕迹", "with signs of disturbance"),
    ("指纹", "with visible print marks"),
    ("掌纹", "with visible palm prints"),
    ("毛发", "with visible hair strands"),
    ("纤维", "with visible fabric fibers"),
    ("泥土", "with soil residue"),
    ("灰尘", "with dust marks"),
    ("毒物", "with toxic residue"),
    ("药物", "with drug residue"),
    ("破碎", "broken"),
    ("碎片", "shattered"),
    ("散落", "scattered"),
    ("划痕", "with scratches"),
    ("撕裂", "torn"),
    ("烧灼", "charred"),
    ("污渍", "stained"),
    ("湿润", "wet surface"),
    ("锈蚀", "rusted"),
    ("裂纹", "cracked"),
    ("折叠", "folded"),
    ("开启", "open"),
    ("关闭", "closed"),
    ("上锁", "locked"),
    ("空的", "empty"),
    ("装满", "full"),
    ("凶器", "tool item"),
    ("生物痕迹", "with trace materials"),
    ("DNA", "with trace materials"),
    ("钝器伤", "with blunt force marks"),
    ("锐器伤", "with sharp force marks"),
    ("牙齿印", "with bite marks"),
    ("抓痕", "with scratch marks"),
    ("淤青", "with bruise marks"),
    ("挫伤", "with contusion marks"),
    ("擦挫伤", "with abrasion marks"),
    ("表皮剥脱", "with abrasion marks"),
    ("皮下出血", "with subcutaneous discoloration"),
    ("创口", "with open wound"),
    ("裂伤", "with laceration"),
    ("砍创", "with chop mark"),
    ("刺创", "with stab mark"),
    ("骨折", "with fracture"),
    ("红肿", "with swelling"),
    ("肿胀", "with swelling"),
    ("皮下淤血", "with subcutaneous hematoma"),
]

_CN_STATE_PATTERNS_MEDIUM: list[tuple[str, str]] = [
    ("喷溅血迹", "with noticeable reddish-brown splash patterns on the surface"),
    ("擦拭状血迹", "with smudged reddish-brown wipe patterns on the surface"),
    ("滴落状血迹", "with distinct drip mark patterns on the surface"),
    ("血迹分布", "with distributed reddish-brown discoloration areas"),
    ("干涸血迹", "with dark brown dried stain marks on the surface"),
    ("氧化血迹", "with dark brown oxidized discoloration marks"),
    ("接触性出血", "with contact transfer stain marks visible"),
    ("鲜血", "with bright red-brown wet stain marks on the surface"),
    ("血迹", "with noticeable reddish-brown stain marks on the surface"),
    ("血痕", "with noticeable reddish-brown stain marks on the surface"),
    ("带血", "with prominent red-brown stains on the surface"),
    ("血", "with visible reddish-brown stain marks"),
    ("唾液印记", "with distinct dark moist stain marks on the surface"),
    ("唾液斑", "with distinct dark moist stain marks on the surface"),
    ("唾液", "with visible dark wet stain marks on the surface"),
    ("精斑", "with visible biological stain marks"),
    ("精液", "with visible biological stain marks"),
    ("凌乱", "in a messy, disarranged state with items displaced and scattered"),
    ("混乱", "in a messy, disarranged state with items displaced and scattered"),
    ("翻倒", "overturned, knocked over and tipped on its side"),
    ("打斗痕迹", "with signs of physical disturbance, items displaced and knocked over"),
    ("挣扎痕迹", "with signs of physical disturbance and displaced items"),
    ("指纹", "with visible fingerprint marks on the surface"),
    ("掌纹", "with visible palm print marks on the surface"),
    ("毛发", "with visible hair strands present"),
    ("纤维", "with visible fabric fibers present"),
    ("泥土", "with soil residue and dirt marks visible"),
    ("灰尘", "with dust and dirt marks visible"),
    ("毒物", "with toxic substance residue visible"),
    ("药物", "with drug substance residue visible"),
    ("破碎", "broken or shattered into pieces"),
    ("碎片", "shattered into small fragments"),
    ("散落", "scattered across the ground"),
    ("划痕", "with visible cut marks or scratches on the surface"),
    ("撕裂", "torn, ripped material visible"),
    ("烧灼", "charred, burnt marks visible"),
    ("污渍", "stained, dirty marks visible"),
    ("湿润", "wet, moist surface visible"),
    ("锈蚀", "rusted, with rust stains visible"),
    ("裂纹", "cracked, with crack lines visible"),
    ("折叠", "folded"),
    ("开启", "in open state"),
    ("关闭", "closed"),
    ("上锁", "locked"),
    ("空的", "empty"),
    ("装满", "full"),
    ("凶器", "sharp tool item"),
    ("生物痕迹", "with biological trace materials visible"),
    ("DNA", "with potential biological trace materials"),
    ("钝器伤", "with blunt force injury marks"),
    ("锐器伤", "with sharp force injury marks, cut marks"),
    ("牙齿印", "with teeth marks, bite marks visible"),
    ("抓痕", "with scratch marks visible"),
    ("淤青", "with bruise marks, contusion visible"),
    ("挫伤", "with contusion, bruising visible"),
    ("擦挫伤", "with abrasion and contusion marks"),
    ("表皮剥脱", "with skin abrasion marks visible"),
    ("皮下出血", "with subcutaneous discoloration visible"),
    ("创口", "with open wound visible"),
    ("裂伤", "with laceration wound visible"),
    ("砍创", "with chop wound mark visible"),
    ("刺创", "with stab wound mark visible"),
    ("骨折", "with bone fracture visible"),
    ("红肿", "with swelling, red and swollen"),
    ("肿胀", "with swelling visible"),
    ("皮下淤血", "with subcutaneous hematoma visible"),
]

_CN_STATE_PATTERNS_DETAILED: list[tuple[str, str]] = [
    ("喷溅血迹", "with prominent bright red blood spatter on the surface, fresh blood splashes clearly visible"),
    ("擦拭状血迹", "with smeared bloodstain patterns, wipe marks from blood clearly visible"),
    ("滴落状血迹", "with dripped bloodstain pattern, round blood drops clearly visible"),
    ("血迹分布", "with blood stain distribution pattern, multiple blood stains visible"),
    ("干涸血迹", "with dark dried bloodstains, oxidized blood residue visible"),
    ("氧化血迹", "with dark brown oxidized blood stains"),
    ("接触性出血", "with contact bleeding marks, transfer bloodstains visible"),
    ("鲜血", "with vivid fresh bright red blood stains, fresh blood wet and glistening"),
    ("血迹", "with prominent vivid red blood stains, fresh blood stains clearly visible on the surface"),
    ("血痕", "with prominent vivid red blood stains, fresh blood stains clearly visible on the surface"),
    ("带血", "covered with prominent vivid red blood stains, fresh blood clearly visible"),
    ("血", "with visible blood stains"),
    ("唾液印记", "with prominent dark saliva stain marks on the surface, saliva residue clearly visible as dark wet patches"),
    ("唾液斑", "with prominent dark saliva stain marks on the surface, saliva residue clearly visible as dark wet patches"),
    ("唾液", "with visible dark saliva stains and moist saliva residue"),
    ("精斑", "with visible biological stain marks"),
    ("精液", "with visible biological stain marks"),
    ("凌乱", "in a chaotic, messy state, items scattered and overturned, signs of struggle visible"),
    ("混乱", "in a chaotic, messy state, items scattered and overturned, signs of struggle visible"),
    ("翻倒", "overturned, knocked over, tipped on its side"),
    ("打斗痕迹", "with clear signs of violent struggle, items displaced and knocked over"),
    ("挣扎痕迹", "with clear signs of struggle and resistance"),
    ("指纹", "with visible fingerprints on the surface"),
    ("掌纹", "with visible palm prints on the surface"),
    ("毛发", "with visible hair fibers"),
    ("纤维", "with visible fabric fibers"),
    ("泥土", "with soil residue and dirt marks visible"),
    ("灰尘", "with dust and dirt marks visible"),
    ("毒物", "with toxic substance residue visible"),
    ("药物", "with drug residue visible"),
    ("破碎", "broken or shattered into pieces"),
    ("碎片", "shattered into fragments"),
    ("散落", "scattered on the ground"),
    ("划痕", "with visible cut marks or scratches on the surface"),
    ("撕裂", "torn, ripped fabric visible"),
    ("烧灼", "charred, burnt marks visible"),
    ("污渍", "stained, dirty marks visible"),
    ("湿润", "wet, moist surface visible"),
    ("锈蚀", "rusted, rust stains visible"),
    ("裂纹", "cracked, crack lines visible"),
    ("折叠", "folded"),
    ("开启", "in open state"),
    ("关闭", "closed"),
    ("上锁", "locked"),
    ("空的", "empty"),
    ("装满", "full"),
    ("凶器", "weapon, murder weapon"),
    ("生物痕迹", "with biological trace evidence visible"),
    ("DNA", "with potential DNA evidence"),
    ("钝器伤", "with blunt force injury marks"),
    ("锐器伤", "with sharp force injury marks, cut marks"),
    ("牙齿印", "with teeth marks, bite marks visible"),
    ("抓痕", "with scratch marks visible"),
    ("淤青", "with bruise marks, contusion visible"),
    ("挫伤", "with contusion, bruising visible"),
    ("擦挫伤", "with abrasion and contusion marks"),
    ("表皮剥脱", "with skin abrasion marks"),
    ("皮下出血", "with subcutaneous bleeding visible"),
    ("创口", "with open wound visible"),
    ("裂伤", "with laceration wound visible"),
    ("砍创", "with chop wound visible"),
    ("刺创", "with stab wound visible"),
    ("骨折", "with bone fracture"),
    ("红肿", "with swelling, red and swollen"),
    ("肿胀", "with swelling visible"),
    ("皮下淤血", "with subcutaneous hematoma visible"),
]

# 默认使用安全版（向后兼容）
_CN_STATE_PATTERNS = _CN_STATE_PATTERNS_SAFE


def _translate_state(state: dict | None, patterns: list[tuple[str, str]] | None = None) -> str:
    """将状态 dict 转为英文描述字符串。支持英文键和中文键两种格式。

    字符串类型的值（如 fingerprints: "张默"）会被保留为 key:value 对，
    确保细节信息（指纹归属、血迹来源等）不丢失。

    patterns: 指定使用的状态映射版本（安全版或详细版）
    """
    if not state:
        return ""
    if patterns is None:
        patterns = _CN_STATE_PATTERNS
    parts = []
    for k, v in state.items():
        if isinstance(v, bool) and v is True:
            adj = _STATE_ADJ_EN.get(k)
            if not adj:
                adj = _match_cn_state(k, patterns=patterns)
            parts.append(adj)
        elif isinstance(v, str):
            cn_match = _match_cn_state(k, patterns=patterns)
            if cn_match != k:
                parts.append(f"{cn_match} ({v})")
            else:
                parts.append(f"{k}: {v}")
        elif isinstance(v, bool) and v is False:
            pass
        else:
            parts.append(f"{k}: {v}")
    return ", ".join(parts)


def _match_cn_state(key: str, patterns: list[tuple[str, str]] | None = None) -> str:
    """用中文关键词匹配英文状态描述。

    patterns: 指定使用的状态映射版本（安全版或详细版）
    """
    if patterns is None:
        patterns = _CN_STATE_PATTERNS
    for cn_kw, en_desc in patterns:
        if cn_kw in key:
            return en_desc
    return key


# ==============================================================================
# 场景级别环境信号聚合（确保地面血迹/混乱等不只依附于单个物证）
# ==============================================================================

# 场景级别环境信号聚合（安全版和详细版）
# (触发关键词集合, 场景级别描述)
_SCENE_LEVEL_SIGNALS_SAFE: list[tuple[set[str], str]] = [
    (
        {"凌乱", "混乱", "翻倒", "打斗痕迹", "挣扎痕迹", "chaotic", "messy", "overturned", "struggle"},
        "The room appears disarranged with items out of their usual placement — "
        "furniture and objects are displaced, scattered, and some are overturned.",
    ),
    (
        {"血迹", "喷溅血迹", "干涸血迹", "滴落状血迹", "擦拭状血迹", "血痕", "鲜血", "带血", "血", "bloody", "blood"},
        "Reddish-brown discoloration marks are visible on the floor — "
        "pooled areas and marked discolorations on floor surfaces.",
    ),
    (
        {"唾液", "唾液印记", "唾液斑", "saliva"},
        "Dark moist patches are visible on floor surfaces and nearby objects.",
    ),
    (
        {"破碎", "碎片", "散落", "broken", "shattered", "scattered"},
        "Broken fragments and scattered debris are present on the floor.",
    ),
    (
        {"指纹", "掌纹", "fingerprints", "palm"},
        "Visible print marks are present on item surfaces and adjacent walls.",
    ),
    (
        {"生物痕迹", "DNA", "毛发", "纤维", "biological", "dna", "hair", "fiber"},
        "Trace evidence materials (hair strands, fabric fibers) are present "
        "in the scene, scattered on the floor and around evidence items.",
    ),
]

_SCENE_LEVEL_SIGNALS_MEDIUM: list[tuple[set[str], str]] = [
    (
        {"凌乱", "混乱", "翻倒", "打斗痕迹", "挣扎痕迹", "chaotic", "messy", "overturned", "struggle"},
        "The room is in a messy, disarranged state — furniture and objects are displaced, "
        "scattered across the floor, and some are overturned. Signs of physical disturbance "
        "are visible throughout the scene, with items knocked over and personal effects spread about.",
    ),
    (
        {"血迹", "喷溅血迹", "干涸血迹", "滴落状血迹", "擦拭状血迹", "血痕", "鲜血", "带血", "血", "bloody", "blood"},
        "Noticeable reddish-brown stain patterns are visible on the floor throughout the scene — "
        "pooled areas, splash patterns, drip marks, and smudged discolorations on floor surfaces. "
        "These stain marks appear as distinct red-brown colorations, clearly noticeable "
        "and not subtle or faint.",
    ),
    (
        {"唾液", "唾液印记", "唾液斑", "saliva"},
        "Distinct dark moist stain patches are visible on floor surfaces and nearby objects — "
        "appearing as clear dark wet areas with a moist sheen, noticeable against the surfaces.",
    ),
    (
        {"破碎", "碎片", "散落", "broken", "shattered", "scattered"},
        "Broken fragments and scattered debris are spread across the floor, "
        "with shattered pieces dispersed around the scene area.",
    ),
    (
        {"指纹", "掌纹", "fingerprints", "palm"},
        "Visible fingerprint and palm print marks are present on item surfaces and adjacent walls.",
    ),
    (
        {"生物痕迹", "DNA", "毛发", "纤维", "biological", "dna", "hair", "fiber"},
        "Biological trace materials (hair strands, fabric fibers, other small particles) "
        "are present in the scene, scattered on the floor and around evidence items.",
    ),
]

_SCENE_LEVEL_SIGNALS_DETAILED: list[tuple[set[str], str]] = [
    (
        {"凌乱", "混乱", "翻倒", "打斗痕迹", "挣扎痕迹", "chaotic", "messy", "overturned", "struggle"},
        "The WHOLE ROOM is in CHAOTIC, MESSY DISARRAY — furniture and items are displaced, "
        "scattered, knocked over, and overturned. Clear SIGNS OF VIOLENT STRUGGLE are visible "
        "throughout the scene. The overall environment looks disheveled and disturbed.",
    ),
    (
        {"血迹", "喷溅血迹", "干涸血迹", "滴落状血迹", "擦拭状血迹", "血痕", "鲜血", "带血", "血", "bloody", "blood"},
        "PROMINENT VIVID RED BLOOD STAINS are clearly visible on the FLOOR throughout the scene — "
        "fresh glistening blood pools, blood spatter patterns, dripped blood drops, and wiped "
        "bloodstain marks on the floor. The blood appears fresh, BRIGHT RED, and HIGHLY VISIBLE — "
        "not subtle or hidden. The floor shows clear evidence of bloodshed.",
    ),
    (
        {"唾液", "唾液印记", "唾液斑", "saliva"},
        "DARK WET SALIVA STAIN MARKS are visible on the floor and surfaces — "
        "saliva residue appears as distinct dark wet patches, highly visible and unmistakable.",
    ),
    (
        {"破碎", "碎片", "散落", "broken", "shattered", "scattered"},
        "Broken fragments and scattered debris are visible on the floor, "
        "items appear smashed and scattered around the scene.",
    ),
    (
        {"指纹", "掌纹", "fingerprints", "palm"},
        "Visible fingerprints and palm prints on the surfaces of items and nearby walls.",
    ),
    (
        {"生物痕迹", "DNA", "毛发", "纤维", "biological", "dna", "hair", "fiber"},
        "Biological trace evidence (hair fibers, fabric fibers, potential DNA) is visible "
        "in the scene, scattered on the floor and around evidence items.",
    ),
]

# 默认使用安全版（向后兼容）
_SCENE_LEVEL_SIGNALS = _SCENE_LEVEL_SIGNALS_SAFE


def _aggregate_scene_conditions(objects: list["SceneObject"], signals: list[tuple[set[str], str]] | None = None) -> list[str]:
    """聚合所有物证的 state/description 字段，生成场景级别的环境描述列表。

    作用：
    - 物证 state 描述的是"这个物证的状态"（如"刀上有血迹"）
    - 场景级别描述的是"整个场景的环境"（如"地面有明显血迹"、"房间混乱"）
    - 即便原文只把"血迹"附着在某个物证上，本函数也将其提升为场景级别特征
      （地面血迹本质上是场景属性，不应只依附于单一物证）

    signals: 指定使用的场景级别信号版本（安全版或详细版）

    Returns:
        list[str]: 场景级别环境描述列表（去重，按 signals 顺序）
    """
    import logging
    logger = logging.getLogger(__name__)

    if signals is None:
        signals = _SCENE_LEVEL_SIGNALS

    all_text: list[str] = []
    for obj in objects:
        if obj.state:
            for k, v in obj.state.items():
                all_text.append(str(k))
                if isinstance(v, str):
                    all_text.append(v)
        if obj.description:
            all_text.append(obj.description)

    combined = "\n".join(all_text)

    conditions: list[str] = []
    matched_signals: set[int] = set()
    for idx, (triggers, desc) in enumerate(signals):
        for trigger in triggers:
            if trigger in combined:
                conditions.append(desc)
                matched_signals.add(idx)
                logger.info(
                    f"[scene_engine] 场景级信号触发 [{trigger}] -> 描述加入 prompt"
                )
                break

    return conditions


# ==============================================================================
# 场景构建时排除的证据类型（不出现在场景图中）
# ==============================================================================

# 这些类型的证据即使被确认，也不会出现在场景图中
SCENE_EXCLUDED_TYPES = {
    "鉴定意见",
    "勘验笔录",
    "书证",      # 书证是纸质文件，不需要放在场景图中
    "电子数据",  # 电子数据是虚拟的，不需要放在场景图中
    "人身检查",  # 人身检查结果用于伤情特写，不出现在场景图中
}


def _is_excluded_from_scene(evidence_type: str) -> bool:
    """判断证据类型是否应该从场景图中排除。

    Args:
        evidence_type: 证据类型字符串

    Returns:
        bool: True 表示应该排除
    """
    if not evidence_type:
        return False
    # 精确匹配
    if evidence_type in SCENE_EXCLUDED_TYPES:
        return True
    # 关键词匹配
    excluded_keywords = ["鉴定", "勘验", "笔录", "报告", "意见书", "鉴定书"]
    for kw in excluded_keywords:
        if kw in evidence_type:
            return True
    return False


# ==============================================================================
# 核心函数
# ==============================================================================

def get_render_position(location_key: str) -> dict:
    """将空间关系键名转为渲染坐标信息。

    如果 location_key 不在 LOCATION_MAP 中，返回默认位置 floor_center。

    Args:
        location_key: 如 "under_bed_left", "on_desk"

    Returns:
        dict: {"area": "...", "zone": "..."}
    """
    if not location_key:
        return dict(DEFAULT_POSITION)
    return dict(LOCATION_MAP.get(location_key, DEFAULT_POSITION))


def _area_to_natural_language(area: str, zone: str) -> str:
    """
    将 area/zone 字符串转为 prompt 中自然语言位置描述（英文）。
    用于 scene_objects_to_rich_prompt() 生成人类可读的位置描述。
    """
    AREA_PHRASES = {
        "lower_left": "on the lower-left area of the floor",
        "lower_right": "on the lower-right area of the floor",
        "lower_center": "on the floor near the center-bottom",
        "upper_left": "on the upper-left area",
        "upper_right": "on the upper-right area",
        "upper_center": "on the upper-center area",
        "center": "in the center of the scene",
        "center_left": "on the left-center area",
        "center_right": "on the right-center area",
        "left_center": "along the left side",
        "right_center": "along the right side",
        "top_center": "near the top center (wall area)",
        "far_left": "in the far left corner",
        "far_right": "in the far right corner",
    }

    ZONE_PHRASES = {
        "bed_area": "near the bed",
        "desk_area": "near the desk",
        "sofa_area": "near the sofa or coffee table",
        "reception_area": "near the reception desk",
        "kitchen_area": "in the kitchen area",
        "bathroom_area": "in the bathroom area",
        "floor_area": "on the floor",
        "wall_area": "against the wall",
        "storage_area": "near the storage or wardrobe",
        "clothing_area": "on or near clothing",
        "door_area": "near the door",
        "window_area": "near the window",
    }

    # 优先用 zone 描述（更语义化），fallback 到 area
    return ZONE_PHRASES.get(zone, AREA_PHRASES.get(area, "somewhere in the scene"))


def _infer_room_type(scene_name: str) -> str:
    """根据场景名称中的关键词推断 base_room_type。"""
    for kw, room_type in ROOM_KEYWORDS.items():
        if kw in scene_name:
            return room_type
    return "bedroom"  # 默认


def _generate_object_id(obj_type: str, description: str, index: int) -> str:
    """生成物品唯一 ID，如 knife_1, phone_2。"""
    # 从描述中提取英文关键词
    type_hint = description[:4] if len(description) <= 8 else description[:3]
    sanitized = "".join(c for c in type_hint if c.isalnum() or c in "_")
    if not sanitized:
        sanitized = obj_type.replace(" ", "_")
    return f"{sanitized}_{index}"


def build_scene_from_evidence(
    evidence_list: list[EvidenceItem],
    scene_name: str = "案发现场",
) -> SceneState:
    """将证据列表转为 Scene State。

    只处理 extractable_evidence，不处理 uncertain 和 non_visualizable。
    自动过滤掉不应出现在场景图中的证据类型（鉴定意见、勘验笔录、书证、电子数据等）。
    自动推断 base_room_type（根据场景名称关键词）。

    Args:
        evidence_list: 来自 EvidenceFilterResult.extractable_evidence
        scene_name: 场景名称

    Returns:
        SceneState: 包含所有可渲染物品的结构化场景状态
    """
    import logging
    logger = logging.getLogger(__name__)

    objects: list[SceneObject] = []
    excluded_count = 0

    for idx, ev in enumerate(evidence_list, 1):
        # 过滤掉不应出现在场景图中的证据类型
        if _is_excluded_from_scene(ev.evidence_type):
            excluded_count += 1
            logger.info(
                f"[build_scene_from_evidence] 排除非场景证据: "
                f"[{ev.evidence_type}] {ev.description[:50]}"
            )
            continue

        loc_key = ev.location or "on_floor"
        render_pos = get_render_position(loc_key)

        obj = SceneObject(
            id=_generate_object_id(ev.evidence_type, ev.description, idx),
            object_type=ev.evidence_type,
            description=ev.description,
            location_key=loc_key,
            render_position=render_pos,
            state=ev.state or {},
            evidence_category="extractable",
        )
        objects.append(obj)

    if excluded_count > 0:
        logger.info(
            f"[build_scene_from_evidence] 共排除 {excluded_count} 项非场景证据"
        )

    base_room = _infer_room_type(scene_name)

    return SceneState(
        scene_id=f"scene_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        scene_name=scene_name,
        base_room_type=base_room,
        objects=objects,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def scene_to_image_prompt(scene: SceneState, case_style: str | None = None) -> str:
    """将 SceneState 转为空白场景生成的英文 Prompt。

    用于空白场景生成（不含物证）。

    Args:
        scene: 已构建的场景状态
        case_style: 可选的 case 级别环境/装修风格描述

    Returns:
        str: 英文 prompt 字符串
    """
    room_desc = _ROOM_TYPE_EN.get(scene.base_room_type, "A generic indoor scene")
    case_prefix = f" The scene is located in: {case_style}." if case_style else ""
    return (
        f"crime scene investigation photo, {room_desc}.{case_prefix}"
        f"empty room with no people, no furniture or minimal furniture, "
        f"forensic style, overhead view, "
        f"professional lighting, 4k, photorealistic"
    )


def object_to_inpaint_prompt(obj: SceneObject) -> str:
    """将单个物品转为 inpainting 的英文 Prompt。

    Args:
        obj: 单个场景物品

    Returns:
        str: 英文 inpainting prompt 字符串
    """
    type_en = _OBJ_TYPE_EN.get(obj.object_type, "evidence item")
    state_str = _translate_state(obj.state)

    parts = ["forensic evidence photo", type_en]
    parts.append(f"a {obj.description}" if obj.description else "(unknown object)")

    if state_str:
        parts.append(state_str)

    parts.append("realistic, forensic investigation style, neutral background")
    return ", ".join(parts)


def evidence_list_to_prompts(
    evidence_list: list[EvidenceItem],
    scene_name: str = "案发现场",
) -> dict:
    """一站式函数：将证据列表直接转为场景 + 全部物品 prompt。

    返回结构：
    {
        "scene_state": SceneState JSON,
        "scene_prompt": str,        # 空白场景 prompt
        "object_prompts": list[dict] # 每个物品的 prompt + id
    }

    Args:
        evidence_list: 来自 EvidenceFilterResult.extractable_evidence
        scene_name: 场景名称

    Returns:
        dict: 包含 scene_state, scene_prompt, object_prompts
    """
    scene = build_scene_from_evidence(evidence_list, scene_name)
    scene_prompt = scene_to_image_prompt(scene)

    object_prompts = []
    for obj in scene.objects:
        object_prompts.append({
            "id": obj.id,
            "prompt": object_to_inpaint_prompt(obj),
            "description": obj.description,
            "render_position": obj.render_position,
        })

    return {
        "scene_state": scene.model_dump(),
        "scene_prompt": scene_prompt,
        "object_prompts": object_prompts,
    }


# ==============================================================================
# Rich Prompt Builder（一场景一图）
# ==============================================================================

# ⭐ DALL·E 3 的安全 prompt 上限
_MAX_PROMPT_CHARS = 3800
_PROMPT_WARNING_THRESHOLD = 8  # 超过此数量的物品时，前端提示教师拆分场景


def scene_objects_to_rich_prompt(
    scene: SceneState,
    case_style: str | None = None,
    provider: str = "dalle",
    detail_level: str = "auto",
) -> tuple[str, list[str]]:
    """
    将场景状态转为图像生成的 Rich Prompt（英文）。

    返回：
    - prompt: str — 最终用于 API 调用的 prompt
    - omitted_items: list[str] — 因超出长度上限而被省略的物品描述列表

    参数：
    - provider: 当前使用的图像生成模型名称
    - detail_level: 细节级别
      - "auto": 根据 provider 自动选择（zenmux→medium, 其他→detailed）
      - "safe": 安全版（极简，确保通过安全过滤）
      - "medium": 中等版（平衡安全和效果）
      - "detailed": 详细版（完整细节）

    核心原则：
    - 代码控制 AI，不允许 AI 自由发挥
    - 所有物品位置来自 SceneState，不推断
    - prompt 不超过 _MAX_PROMPT_CHARS 字符
    - ⭐ 不丢弃任何物品：超出上限时使用紧凑格式而非省略
    """
    import logging
    logger = logging.getLogger(__name__)

    # 确定细节级别
    if detail_level == "auto":
        if provider.lower() == "zenmux":
            detail_level = "medium"
        else:
            detail_level = "detailed"

    # 根据级别选择 prompt 模板
    level_map = {
        "safe":     (_CN_STATE_PATTERNS_SAFE,     _SCENE_LEVEL_SIGNALS_SAFE),
        "medium":   (_CN_STATE_PATTERNS_MEDIUM,   _SCENE_LEVEL_SIGNALS_MEDIUM),
        "detailed": (_CN_STATE_PATTERNS_DETAILED, _SCENE_LEVEL_SIGNALS_DETAILED),
    }
    state_patterns, scene_signals = level_map.get(
        detail_level,
        (_CN_STATE_PATTERNS_MEDIUM, _SCENE_LEVEL_SIGNALS_MEDIUM),
    )
    logger.info(
        f"[scene_engine] 场景图 prompt 级别: {detail_level} "
        f"(provider: {provider}, 物品数: {len(scene.objects)})"
    )

    room_desc = _ROOM_TYPE_EN.get(scene.base_room_type, "a room interior")

    case_style_prefix = ""
    if case_style:
        case_style_prefix = (
            f" The interior setting is: {case_style}."
        )

    # 场景级环境条件
    scene_conditions = _aggregate_scene_conditions(scene.objects, signals=scene_signals)
    env_section = ""
    if scene_conditions:
        env_section = (
            "\nSCENE ENVIRONMENT DETAILS:\n"
            + "\n".join(f"- {cond}" for cond in scene_conditions)
            + "\n"
        )

    # 根据级别选择前缀和后缀
    if detail_level in ("safe", "medium"):
        prefix = (
            f"Educational training photograph, {room_desc}."
            f"{case_style_prefix}"
            f" Professional documentation photography, "
            f"wide-angle view, realistic lighting, 4k high detail, no people, no text. "
            f"The following items are present in the room:\n"
            f"{env_section}"
        )
        if detail_level == "safe":
            suffix = (
                "\nRENDERING INSTRUCTIONS:\n"
                "- Every item listed above must be clearly visible and properly placed.\n"
                "- All surface details (marks, discolorations, scratches, breaks, "
                "wet areas) must be clearly visible on the corresponding items.\n"
                "- Red-brown marks must appear as distinct colorations on surfaces.\n"
                "- If items show signs of being disturbed, the scene should reflect "
                "a disarranged state with items out of their usual order.\n"
                "- Place each item at the described position with its described condition.\n"
                "- Sharp focus, professional quality, educational documentation photograph."
            )
        else:
            suffix = (
                "\nRENDERING INSTRUCTIONS:\n"
                "- Every item listed above must be clearly visible and properly placed.\n"
                "- All surface details (stains, marks, discolorations, scratches, "
                "breaks, wet areas) must be clearly visible on the corresponding items.\n"
                "- Reddish-brown stain marks must appear as distinct, noticeable "
                "colorations on surfaces — not subtle or faint.\n"
                "- Dark moist patches must be clearly visible as distinct wet areas.\n"
                "- If items show signs of being disturbed, the overall scene should "
                "reflect a messy, disarranged state with items displaced and scattered.\n"
                "- The SCENE ENVIRONMENT DETAILS above must all be clearly visible, "
                "especially stain patterns on the floor and surfaces.\n"
                "- Place each item at the described position with its described condition.\n"
                "- Sharp focus, professional quality, educational training photograph."
            )
    else:
        prefix = (
            f"Forensic crime scene investigation photo, {room_desc}."
            f"{case_style_prefix}"
            f" Crime scene tape visible. Professional forensic photography, "
            f"overhead or wide-angle view, realistic, 4k, no people, no text."
            f"{env_section}"
            f" The following evidence items are present:\n"
        )
        suffix = (
            "\nCRITICAL RENDERING RULES:\n"
            "- EVERY item listed above MUST be clearly visible in the image.\n"
            "- ALL state details (blood, saliva, fingerprints, stains, tears, breaks) "
            "MUST be HIGHLY VISIBLE and clearly rendered on the items — vivid red "
            "bloodstains must look bright red and prominent, not subtle.\n"
            "- Saliva stains must be visible as dark wet patches.\n"
            "- If any item has a 'chaotic' or 'messy' or 'overturned' state, the "
            "overall scene should reflect signs of struggle, with items displaced, "
            "scattered, and the environment in disarray.\n"
            "- Place each item exactly as described with the described state and position.\n"
            "- The SCENE ENVIRONMENT section above is NON-NEGOTIABLE — blood pools, "
            "saliva stains, and chaotic state must be clearly visible at the scene level, "
            "especially on the FLOOR (not only on individual items).\n"
            "- Forensic evidence style, high detail, realistic lighting, sharp focus "
            "on all evidence items."
        )

    fixed_length = len(prefix) + len(suffix)
    available_chars = _MAX_PROMPT_CHARS - fixed_length

    # 逐条构建物品描述
    item_lines = []
    omitted_items = []
    current_length = 0

    for i, obj in enumerate(scene.objects):
        state_desc = _translate_state(obj.state, patterns=state_patterns)
        if state_desc:
            state_desc = f", {state_desc}"

        position_desc = _area_to_natural_language(
            obj.render_position.get("area", "center"),
            obj.render_position.get("zone", "floor_area"),
        )

        # ⭐ 物品描述增强：匹配常见类型，添加详细英文描述
        item_name_en = _enrich_item_description(obj.description)

        # 完整格式（使用增强的英文描述）
        full_line = f"{i+1}. {item_name_en}{state_desc} — {position_desc}\n"
        # 紧凑格式（空间不足时使用）
        compact_line = f"{i+1}. {obj.description} — {position_desc}\n"

        if current_length + len(full_line) <= available_chars:
            item_lines.append(full_line)
            current_length += len(full_line)
        elif current_length + len(compact_line) <= available_chars:
            # 空间不足以用完整格式，使用紧凑格式
            item_lines.append(compact_line)
            current_length += len(compact_line)
            logger.info(
                f"[scene_engine] 物品 '{obj.description}' 使用紧凑格式 "
                f"(场景: {scene.scene_name}, 节省 {len(full_line) - len(compact_line)} 字符)"
            )
        else:
            # 连紧凑格式也放不下 — 记录为省略（极少发生）
            omitted_items.append(obj.description)
            logger.warning(
                f"[scene_engine] Prompt 长度超限，物品被省略: {obj.description} "
                f"(场景: {scene.scene_name}, 当前长度: {fixed_length + current_length})"
            )

    # 如果有被省略的物品，加注
    omit_note = ""
    if omitted_items:
        omit_note = (
            f"\n[WARNING: {len(omitted_items)} additional items could not fit "
            f"in the prompt. Consider splitting this scene.]"
        )

    final_prompt = prefix + "".join(item_lines) + suffix + omit_note

    # 最终安全检查（极少触发，只有 prompt 在紧凑格式后仍超出才截断）
    if len(final_prompt) > _MAX_PROMPT_CHARS:
        final_prompt = final_prompt[:_MAX_PROMPT_CHARS]
        logger.error(f"[scene_engine] Prompt 超出硬上限，已强制截断")

    return final_prompt, omitted_items

    return final_prompt, omitted_items


def get_scene_prompt_warning(scene: SceneState) -> dict:
    """
    检查场景物品数量，返回给前端的提示信息。
    用于在生成前告知教师是否需要拆分场景。
    """
    count = len(scene.objects)
    if count > _PROMPT_WARNING_THRESHOLD:
        return {
            "should_warn": True,
            "message": (
                f"当前场景有 {count} 项物证，超过建议上限（{_PROMPT_WARNING_THRESHOLD}）。"
                f"建议将场景拆分为多个，以确保所有物品都能在图中呈现。"
            ),
            "item_count": count,
            "threshold": _PROMPT_WARNING_THRESHOLD,
        }
    return {"should_warn": False, "item_count": count}


# ==============================================================================
# 场景自动建议
# ==============================================================================


def suggest_scenes_from_text(raw_text: str) -> list[dict]:
    """
    扫描案件文本，根据 ROOM_KEYWORDS 建议场景列表。
    返回去重、按出现顺序排列的建议。
    无匹配时返回一个默认「案发现场」建议。

    返回格式：
    [
        {"name": "卧室", "room_type": "bedroom", "reason": "文本中出现了关键词「卧室」"},
        {"name": "客厅", "room_type": "living_room", "reason": "文本中出现了关键词「客厅」"},
    ]
    """
    suggestions = []
    seen_room_types = set()

    for keyword, room_type in ROOM_KEYWORDS.items():
        if keyword in raw_text and room_type not in seen_room_types:
            suggestions.append({
                "name": keyword,
                "room_type": room_type,
                "reason": f"文本中出现了关键词「{keyword}」",
            })
            seen_room_types.add(room_type)

    # 按在文本中首次出现的位置排序
    suggestions.sort(key=lambda s: raw_text.find(s["name"]))

    # 无匹配时给默认建议
    if not suggestions:
        suggestions = [{
            "name": "案发现场",
            "room_type": "unknown",
            "reason": "文本中未检测到具体场景关键词，建议使用默认场景名称",
        }]

    return suggestions


# ==============================================================================
# 测试用例
# ==============================================================================

if __name__ == "__main__":
    import json

    # 模拟 evidence_filter 输出
    test_evidences = [
        EvidenceItem(
            evidence_type="物证",
            description="带血水果刀",
            location="under_bed_left",
            state={"bloody": True},
            source_quote="床下发现一把带血的水果刀（已提取，编号物证1）",
        ),
        EvidenceItem(
            evidence_type="物证",
            description="破碎屏幕的手机",
            location="under_bed_right",
            state={"broken": True},
            source_quote="旁边有一部破碎屏幕的手机（物证2）",
        ),
    ]

    print("=" * 60)
    print("Scene State Engine — 测试用例")
    print("=" * 60)
    print(f"\n输入证据: {len(test_evidences)} 条\n")

    # 单步测试
    for ev in test_evidences:
        pos = get_render_position(ev.location or "on_floor")
        print(f"  {ev.description}")
        print(f"    location_key: {ev.location}")
        print(f"    render_position: {pos}")
        print()

    # 构建场景
    scene = build_scene_from_evidence(test_evidences, scene_name="卧室")
    print(f"Scene ID: {scene.scene_id}")
    print(f"Scene Name: {scene.scene_name}")
    print(f"Room Type: {scene.base_room_type}")
    print(f"Objects: {len(scene.objects)}")
    print()

    # 完整 JSON 输出
    print("=" * 60)
    print("完整 SceneState JSON:")
    print("=" * 60)
    print(scene.model_dump_json(indent=2, ensure_ascii=False))
    print()

    # 图像 prompt
    print("=" * 60)
    print("空白场景 Prompt:")
    print(scene_to_image_prompt(scene))
    print()

    print("=" * 60)
    print("物品 Inpaint Prompts:")
    for obj in scene.objects:
        prompt = object_to_inpaint_prompt(obj)
        print(f"  [{obj.id}] {prompt}")

    print()
    print("=" * 60)
    print("一站式输出 (evidence_list_to_prompts):")
    result = evidence_list_to_prompts(test_evidences, scene_name="卧室")
    print(json.dumps(result, indent=2, ensure_ascii=False))
