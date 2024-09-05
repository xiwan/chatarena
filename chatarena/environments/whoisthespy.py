import random
import re
from typing import Dict, List, Union

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env

DEFAULT_TOPIC_CODES = {
    "水果": [
        "苹果",
        "香蕉",
        "橘子",
        "葡萄",
        "草莓",
        "菠萝",
        "芒果",
        "西瓜",
    ],
    "动物": [
        "狮子",
        "大象",
        "长颈鹿",
        "猴子",
        "斑马",
        "狮子",
        "熊",
        "袋鼠",
    ],
    "运动": [
        "足球",
        "篮球",
        "网球",
        "乒乓球",
        "游泳",
        "自行车",
        "排球",
        "高尔夫",
    ],
    "国家": [
        "美国",
        "加拿大",
        "巴西",
        "英国",
        "法国",
        "德国",
        "日本",
        "澳大利亚",
    ],
}

def ParseJson(text):
    # 使用正则表达式查找 {} 之间的内容
    json_pattern = re.compile( r'{[\s\S]*?}') 
    json_strings = re.findall(json_pattern, text)
    return json_strings

@register_env
class Whoisthespy(Environment):
    type_name = "whoisthespy"

    def __init__(
        self,
        player_names: List[str],
        topic_codes: Dict[str, List[str]] = None,
        **kwargs,
    ):
        super().__init__(player_names=player_names, topic_codes=topic_codes, **kwargs)

        if topic_codes is None:
            topic_codes = DEFAULT_TOPIC_CODES
        self.topic_codes = topic_codes

        # The "state" of the environment is maintained by the message pool
        self.message_pool = MessagePool()

        # Randomly sample a topic, code and chameleon player
        self.topic = None
        self.code = None
        self.chameleon_name = None
        self.non_chameleon_names = None

        # Game states
        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "give clues"  # "give clues", "accuse", "guess"
        self._players_votes = None
        self._initialized = False

        self.reset()  # To initialize the game (select topic, code, chameleon)

    def get_next_player(self) -> str:
        """Get the next player."""
        if self._current_phase != "guess":
            return self.player_names[self._next_player_idx]
        else:
            return self.chameleon_name

    def reset(self):
        """Sample topic, code and chameleon code."""
        self.topic = random.choice(list(self.topic_codes.keys()))
        self.code = random.choice(self.topic_codes[self.topic])
        self.chameleon_name = random.choice(self.player_names)
        self.non_chameleon_names = [
            name for name in self.player_names if name != self.chameleon_name
        ]

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "give clues"

        self.message_pool.reset()

        self._moderator_speak(f"游戏开始! 词语为: {self.topic}")
        self._moderator_speak(
            f"你们不是卧底. 词语为: {self.code}",
            visible_to=self.non_chameleon_names,
        )
        self._moderator_speak("你是卧底!", visible_to=self.chameleon_name)
        self._moderator_speak(
            "你们可以简要地描绘该词语 (不要直接给出词语). "
            f"不要重复前面的内容，我们从玩家 {self.player_names[0]} 开始."
        )
        self._current_turn = 1

        self._players_votes = {name: 0 for name in self.player_names}

        self._initialized = True
        init_timestep = TimeStep(
            observation=self.get_observation(),
            reward=self.get_zero_rewards(),
            terminal=False,
        )

        return init_timestep

    def print(self):
        self.message_pool.print()

    def get_observation(self, player_name=None) -> List[Message]:
        """Get observation for the player."""
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            return self.message_pool.get_visible_messages(
                player_name, turn=self._current_turn
            )

    def _text2vote(self, text) -> str:
        """Convert text to vote, return a player's name."""
        # lower = text.lower().replace("[", "").replace("]", "").replace(".", "")
        text = text.lower()
        print(text)
        for name in self.player_names:
            candidates = [
                name.lower(),
                name.lower().replace(" ", ""),
                name.lower().replace(" ", "_"),
            ]
            if any([candidate in text for candidate in candidates]):
                return name
        return ""
    
    def _parse2vote(self, text) -> str:
        votejson = ParseJson(text)
        print(votejson)
        for name in self.player_names:
            candidates = [
                name.lower(),
                name.lower().replace(" ", ""),
                name.lower().replace(" ", "_"),
            ]
            if any([candidate in text for candidate in candidates]):
                return name
        return ""

    def _is_true_code(self, text) -> bool:
        """Check whether the text is the true code."""
        # Get the word enclosed by quote marks with regex
        pattern = r"\"(.+?)\""
        match = re.search(pattern, text)
        if match:
            return match.group(1).lower().replace(" ", "") == self.code.lower().replace(
                " ", ""
            )
        else:
            # if no quote marks, check whether the last k words match the code
            words = text.split()
            if len(words) >= len(self.code.split()):
                guessed_term = (
                    "".join(words[-len(self.code.split()) :]).lower().replace(".", "")
                )
                return guessed_term == self.code.lower().replace(" ", "").replace(
                    ".", ""
                )
            else:
                return False

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        """Moderator say something."""
        message = Message(
            agent_name="Moderator",
            content=text,
            turn=self._current_turn,
            visible_to=visible_to,
        )
        self.message_pool.append_message(message)

    def get_rewards(self, chameleon_win: bool) -> Dict[str, float]:
        """Get rewards for each player."""
        rewards = {}
        for name in self.player_names:
            # The winner gets 1, the loser gets 0
            rewards[name] = float((name == self.chameleon_name) == chameleon_win)

        return rewards

    def is_terminal(self) -> bool:
        """Check if the conversation is over."""
        # If the last message is the signal, then the conversation is over
        if self.message_pool.last_message.content.startswith(
            SIGNAL_END_OF_CONVERSATION
        ):
            return True

    def step(self, player_name: str, action: str) -> TimeStep:
        """
        Step function that is called by the arena.

        Args:
            player_name: the name of the player that takes the action
            action: the action that the agents wants to take
        """
        # If not initialized, reset the environment
        if not self._initialized:
            self.reset()

        # self.message_pool.print()
        # print(f"Chameleon: {self.chameleon_name}, Code: {self.code}, Topic: {self.topic}")
        assert (
            player_name == self.get_next_player()
        ), f"Wrong player! It is {self.get_next_player()} turn."
        if self._current_phase == "give clues":
            message = Message(
                agent_name=player_name, content=action, turn=self._current_turn
            )
            self.message_pool.append_message(message)

            # Update the counters
            self._current_turn += 1
            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
            else:
                self._next_player_idx = 0
                self._current_phase = "accuse"
                self._moderator_speak(
                    "给其他玩家投票 (不包括自己),每人只能投一票, 看谁是卧底. "
                    "不要给自己投票."
                )
                self._current_turn += 1

            timestep = TimeStep(
                observation=self.get_observation(),
                reward=self.get_zero_rewards(),
                terminal=False,
            )  # Return all the messages
        elif self._current_phase == "accuse":
            message = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to=[player_name],
            )
            self.message_pool.append_message(message)
            vote = self._text2vote(action)
            if vote in self.player_names:
                self._players_votes[vote] += 1

            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
                rewards = self.get_zero_rewards()
                terminal = False
            else:
                # print(self._players_votes)
                accuse_correct, even_vote = True, False
                max_vote_player = max(self._players_votes, key=self._players_votes.get)
                # detach if other players has the same number of votes
                for name, vote in self._players_votes.items():
                    if (
                        name != max_vote_player
                        and vote == self._players_votes[max_vote_player]
                    ):
                        accuse_correct, even_vote = False, True
                if max_vote_player != self.chameleon_name:
                    accuse_correct = False

                if not accuse_correct:
                    if even_vote:
                        self._moderator_speak(
                            f"平票了. 指控阶段无效. "
                            f"{self.chameleon_name} 是卧底. {self.chameleon_name} 赢得了比赛!"
                        )
                    else:
                        self._moderator_speak(
                            f"最多得票者为 {max_vote_player}. 指控失败. "
                            f"{self.chameleon_name} 是卧底. {self.chameleon_name} 赢得了比赛!"
                        )
                    rewards = self.get_rewards(chameleon_win=True)
                    terminal = True
                else:
                    self._moderator_speak(
                        f"指控阶段成功! {self.chameleon_name} 是卧底! "
                        f"接下来让 {self.chameleon_name} 猜测词语. "
                        '你应该说：我猜测词语是 "..."'
                    )
                    self._current_phase = "guess"
                    rewards = self.get_zero_rewards()
                    terminal = False
                self._current_turn += 1

            timestep = TimeStep(
                observation=self.get_observation(), reward=rewards, terminal=terminal
            )
        elif self._current_phase == "guess":
            message = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to=player_name,
            )
            self.message_pool.append_message(message)
            if self._is_true_code(action):
                self._moderator_speak(
                    f"{player_name} 猜测成功! 词语为 {self.code}. "
                    f"{self.chameleon_name} won!"
                )
                rewards = self.get_rewards(chameleon_win=True)
            else:
                self._moderator_speak(
                    f"{player_name} 猜测失败! 词语为 {self.code}. "
                    f"{self.non_chameleon_names} won!"
                )
                rewards = self.get_rewards(chameleon_win=False)
            timestep = TimeStep(
                observation=self.get_observation(), reward=rewards, terminal=True
            )
        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")

        # Check if the player signals the end of the conversation
        if self.is_terminal():
            timestep.terminal = True

        return timestep
