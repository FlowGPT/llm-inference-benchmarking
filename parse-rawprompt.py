import json
import os
import sys
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize

# 如果是第一次使用nltk, 可能需要下载punkt包
nltk.download('punkt')
nltk.download('punkt_tab')

def count_tokens(text):
    tokens = word_tokenize(text)
    return len(tokens)

class Stat4Conv:
    def __init__(self, convid, available_msg, total_msg, timestamp, token_num_available, total_tokens,system_tokens) -> None:
        self.convid = convid
        self.available_msg = available_msg
        self.total_msg = total_msg
        self.timestamp = timestamp
        self.token_num_available = token_num_available
        self.total_tokens = total_tokens
        self.system_tokens = system_tokens
        self.msg_available_rate = float(available_msg) / total_msg
        self.full_cover = False
        if self.msg_available_rate >= 1:
            self.full_cover = True

    def __str__(self) -> str:
        return f"""{self.timestamp} ConversationID: {self.convid}, Available Messages: {self.available_msg}, Total Messages: {self.total_msg}, Available Rate: {self.msg_available_rate:.2f}, Available Tokens: {self.token_num_available}, Total Tokens: {self.total_tokens} System Tokens: {self.system_tokens} Full Cover: {self.full_cover}"""

def analyze_rawprompt(convid, messages, sysprompt_len, limit, ts):
    print(f"Analyzing conversationId: {convid}, system prompt length: {sysprompt_len}, limit: {limit}, timestamp: {ts}")
    current_tokens_len = sysprompt_len
    token_num_available = sysprompt_len
    msg_num_available = 0
    total_msg_len = len(messages)
    for message in messages:
        mesg = message['message_data']
        content = mesg['content']
        tokens_num = count_tokens(content)
        current_tokens_len += tokens_num
        if current_tokens_len < limit:
            msg_num_available += 1
            token_num_available += tokens_num

    return Stat4Conv(convid, msg_num_available, total_msg_len, ts, token_num_available, current_tokens_len,sysprompt_len)

def summary_stat(stat_list):
    total_available_msg = sum(stat.available_msg for stat in stat_list)
    total_msg = sum(stat.total_msg for stat in stat_list)
    total_tokens_available = sum(stat.token_num_available for stat in stat_list)
    total_tokens = sum(stat.total_tokens for stat in stat_list)
    system_tokens = sum(stat.system_tokens for stat in stat_list)

    msg_available_rate = float(total_available_msg) / total_msg
    total_convs = len(stat_list)
    total_full_cover = sum(1 for stat in stat_list if stat.full_cover)
    full_cover_rate = float(total_full_cover) / total_convs
    tokens_available_rate = float(total_tokens_available) / total_tokens
    total_system_tokens = sum(stat.system_tokens for stat in stat_list)
    system_tokens_rate = float(system_tokens) / total_tokens

    print("total convs num:", total_convs)
    print(f"Total Available Messages: {total_available_msg}, Total Messages: {total_msg}, rate: {msg_available_rate:.2f}")
    print(f"Total Tokens Available: {total_tokens_available}, Total Tokens: {total_tokens}, rate: {tokens_available_rate:.2f}")
    print(f"Total System Tokens: {total_system_tokens}, rate: {system_tokens_rate:.2f}")
    print(f"full cover conv {total_full_cover} out of {total_convs}, rate: {full_cover_rate:.2f}")

def get_prefix_before_marker(text, marker='[/INST]'):
    index = text.find(marker)
    if index == -1:
        raise ValueError("not found the system prompt mark")
    return text[:index]

def get_files_with_prefix(directory, prefix):
    if not os.path.isdir(directory):
        raise ValueError(f"目录不存在: {directory}")

    matched_files = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.startswith(prefix):
                matched_files.append(entry.path)  # 使用 entry.path 获取完整路径

    print(matched_files)
    return matched_files

conversation2infos = defaultdict(list)
promptidset = set()

def read_files_analysis(files_list):
    error_count = 0
    conversations = set()
    all_stats=[]

    for file_path in files_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                covid = data["conversationId"]
                if covid in conversations:
                    print("skip dup conversationId", covid)
                    continue
                conversations.add(covid)
                if "rawInput" in data:
                    try:
                        raw_input = data["rawInput"]
                        system_prompt_id = data["promptId"]
                        time_stamp = data["messages"][-1]['message_data']['createdAt']
                        all_messages = data["messages"]
                        print(f"Processing conversationId: {covid}")
                    except Exception as e:
                        print(f"Error processing conversationId {covid}: {e}")
                        error_count += 1
                        continue
                    system_prompt = get_prefix_before_marker(raw_input)
                    stat_item = analyze_rawprompt(covid, all_messages, count_tokens(system_prompt), 7500, time_stamp)
                    print(stat_item)
                    all_stats.append(stat_item)
                    promptidset.add(system_prompt_id)
                else:
                    raise ValueError("No rawInput found in this line for {covid}.")
            print(f"finished processing {file_path} and error count is {error_count}")
            print(f"Total conversations processed: {len(conversations)}")

    print(f"sys promptid len {len(promptidset)}")
    summary_stat(all_stats)
    
if __name__ == "__main__":
    rawprompt_dir = sys.argv[1]
    prefix_name= sys.argv[2]

    files_list = get_files_with_prefix(rawprompt_dir, prefix_name)
    read_files_analysis(files_list)