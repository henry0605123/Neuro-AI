"""Traditional-Chinese mini corpus for the synthetic ADOS-like dialogue.

Scope note (also recorded in DEVIATIONS.md D6): tone labels here are *internally
consistent*, not linguistically authoritative. Neutral tone (0) is assigned to a short
list of particles/suffixes we are confident about; every other character gets a stable
pseudo-tone from a deterministic hash. That is sufficient and honest for a smoke test --
the renderer and the ground truth use the same label, so tone-accuracy and
tone-consistency features are exercised exactly as they would be on real data. Replacing
`tone_of` with a real pronunciation dictionary is a drop-in change.

Pool size matters and is not cosmetic: with only 3-5 answers per topic, ordinary answers
repeat several times per session and swamp the F3 stereotyped-phrase count (measured:
rho = -0.34, i.e. the wrong sign). Roughly a dozen answers per topic makes ordinary
repetition rare, so an n-gram repeated 3+ times really is idiosyncratic.
"""

from __future__ import annotations

import hashlib

__all__ = ["TOPICS", "QUESTIONS", "YES_NO_QUESTIONS", "ANSWERS", "STEREOTYPED_PHRASES",
           "tone_of", "tones_for", "strip_to_syllables"]

# Particles and suffixes that genuinely carry the neutral tone in Mandarin.
NEUTRAL_TONE_CHARS = set("的了嗎呢吧子們著麼個嘛啦喔耶")

TOPICS = ("picture", "story", "play", "friends", "emotion")

QUESTIONS: dict[str, tuple[str, ...]] = {
    "picture": (
        "你看到這張圖片裡面有什麼",
        "圖片裡的小朋友正在做什麼呢",
        "你覺得他們為什麼會在那個地方",
        "這張圖片讓你想到了什麼事情",
        "圖片裡的天氣看起來是怎麼樣",
        "圖片的角落還有什麼東西呢",
        "你覺得圖片裡的人心情如何",
        "如果你也在圖片裡你會做什麼",
    ),
    "story": (
        "你可以說一個故事給我聽嗎",
        "後來又發生了什麼樣的事情",
        "這個故事裡面的主角是誰呢",
        "你覺得這個故事的結局是什麼",
        "故事的一開始是怎麼樣開始的",
        "主角遇到困難的時候怎麼辦",
        "你最喜歡故事裡的哪一個部分",
        "這個故事想要告訴我們什麼",
    ),
    "play": (
        "我們來玩假裝的遊戲好不好呢",
        "你想要當裡面的哪一個角色",
        "這個玩具還可以拿來做什麼用",
        "你平常最喜歡玩什麼樣的遊戲",
        "我們可以用這些積木蓋什麼",
        "假裝這是一艘船我們要去哪裡",
        "你想邀請誰一起來玩這個遊戲",
        "遊戲的規則你想怎麼樣訂呢",
    ),
    "friends": (
        "你在學校裡面有沒有好朋友呢",
        "你們平常下課會一起做什麼",
        "什麼樣子的人才算是好朋友",
        "你怎麼認識你最好的朋友的",
        "跟朋友吵架的時候你會怎麼做",
        "你會怎麼樣邀請別人一起玩",
        "朋友需要幫忙的時候你會做什麼",
        "你覺得自己是一個怎樣的朋友",
    ),
    "emotion": (
        "什麼樣的事情會讓你覺得開心",
        "你難過的時候通常會怎麼辦呢",
        "生氣的時候身體會有什麼感覺",
        "你會怎麼知道別人不開心呢",
        "害怕的時候你會想要做什麼",
        "什麼事情最近讓你覺得很興奮",
        "你會怎麼樣讓自己冷靜下來",
        "有沒有什麼事情讓你很緊張",
    ),
}

# Yes/no questions must be excluded from the G1 relevance pairing (audit P0-F1).
YES_NO_QUESTIONS: tuple[str, ...] = (
    "你喜歡吃冰淇淋嗎",
    "你今天有沒有睡午覺",
    "這個是不是紅色的呢",
    "你要不要喝一點水",
    "你會不會自己穿鞋子",
    "這個東西你看過嗎",
)

ANSWERS: dict[str, tuple[str, ...]] = {
    "picture": (
        "有一個小朋友在公園裡面玩球",
        "他們在草地上面跑來跑去",
        "今天的天氣很好太陽很大",
        "旁邊還有一隻小狗在看著他們",
        "我覺得他們玩得非常的開心",
        "圖片的左邊有一棵很大的樹",
        "有兩個人坐在長椅上面聊天",
        "天空上面飄著白白的雲",
        "遠遠的地方好像有一座山",
        "有一個媽媽推著嬰兒車經過",
        "地上有掉下來的黃色葉子",
        "我看到有人在放一隻風箏",
        "他們的臉上都笑得很甜",
        "旁邊的路上停了一台腳踏車",
    ),
    "story": (
        "從前有一隻小兔子住在森林裡",
        "後來牠遇到了一個新的朋友",
        "他們兩個一起找到了回家的路",
        "最後大家都過得很快樂",
        "有一天下起了很大的雨",
        "小兔子的籃子突然掉到河裡",
        "一隻老烏龜出來幫忙牠",
        "牠們走過一座很長的木橋",
        "天黑了牠們就在山洞裡休息",
        "第二天早上太陽又出來了",
        "小兔子把水果分給大家吃",
        "媽媽在門口一直等著牠回來",
        "我最喜歡他們過橋的那一段",
        "這個故事說朋友要互相幫忙",
    ),
    "play": (
        "我想要當開火車的那個司機",
        "這個東西可以當成一艘大船",
        "我們可以蓋一座很高的城堡",
        "我要當醫生幫娃娃打針",
        "假裝這裡是我們的祕密基地",
        "我們可以開店賣好吃的東西",
        "這個積木可以變成一條隧道",
        "我想去很遠很遠的小島探險",
        "先輪流丟骰子再往前走幾步",
        "我要邀請班上的同學一起來",
        "這條線後面就不可以跨過去",
        "我們把椅子排成一列當車廂",
        "假裝外面正在下很大的雪",
        "誰先到終點誰就可以先選",
    ),
    "friends": (
        "我在班上有兩個很好的朋友",
        "我們下課的時候會一起玩積木",
        "好朋友就是願意幫忙你的人",
        "我們是坐在隔壁才認識的",
        "吵架以後我會先跟他說對不起",
        "我會走過去問他要不要一起玩",
        "他跌倒的時候我會扶他起來",
        "我覺得我是很願意分享的人",
        "我們常常一起在操場上跑步",
        "他上次把餅乾分給我吃",
        "我們喜歡同一個卡通的角色",
        "有時候我們也會意見不一樣",
        "老師說我們要學會輪流",
        "放學的時候我們會一起走",
    ),
    "emotion": (
        "去公園玩的時候我覺得很開心",
        "我難過的時候會去找媽媽抱抱",
        "生氣的時候胸口會悶悶的",
        "看到別人哭我就知道他不開心",
        "害怕的時候我會躲到棉被裡",
        "下個星期要去動物園我好興奮",
        "我會深呼吸然後數到十",
        "上台講話之前我會很緊張",
        "拿到新的玩具我會笑出來",
        "被誤會的時候我會覺得委屈",
        "我會跟老師說我心裡的感覺",
        "聽到喜歡的歌心情就變好了",
        "肚子餓的時候我容易不耐煩",
        "看到朋友難過我也會不舒服",
    ),
}

# Per-subject idiosyncratic phrases (audit P0-F3). "霹靂高" is the clinician's own example.
# Long enough to segment into several words: a three-character phrase yields at most one
# bigram, so F3's n-gram count (n=2..5) had almost nothing to detect and the feature was
# uncorrelated with the injected stereotypy rate (measured rho = -0.02).
STEREOTYPED_PHRASES: tuple[str, ...] = (
    "霹靂高霹靂無敵霹靂高",
    "轟隆轟隆火車要開了轟隆",
    "紅色的車子不可以碰紅色的車子",
    "不可以這樣子不可以這樣子喔",
    "快快快快走快快快快走",
    "圓圓的球球圓圓的球球轉",
    "電梯要關門了請小心電梯要關門了",
    "一二三四五六七八九十一二三",
)


def tone_of(char: str) -> int:
    """Stable tone label for one character. 0 = neutral, 1-4 = lexical tones."""
    if char in NEUTRAL_TONE_CHARS:
        return 0
    digest = hashlib.blake2s(char.encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, "little") % 4 + 1


def strip_to_syllables(text: str) -> list[str]:
    """One Han character == one Mandarin syllable. Punctuation and spaces are dropped."""
    return [c for c in text if "一" <= c <= "鿿"]


def tones_for(text: str) -> list[int]:
    return [tone_of(c) for c in strip_to_syllables(text)]
