# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a simple multi-turn dataset for testing
"""

import argparse
import os

import pandas as pd




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/multiturn")
    parser.add_argument("--file_path", default="./projectnb/rlhf/mingyuc/verl_github/verl/data/maze_mt/train100000.parquet")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()
    df_mt = pd.read_parquet(args.file_path)
    if 'extra_info' in df_mt.columns:
        chat_list_mt = df_mt['extra_info'].apply(lambda x: x['chat'] if isinstance(x, dict) and 'chat' in x else []).tolist()
    elif 'messages' in df_mt.columns:
        chat_list_mt = df_mt['messages'].tolist()
    else:
        # If the structure is different, try to find the chat data in other columns
        print("Available columns in df_mt:", df_mt.columns.tolist())
        # You may need to adjust this based on the actual structure
        chat_list_mt = []

    # Create example conversations
    conversations = []

    # Conversation 1

    for chat in chat_list_mt:
        # Create a conversation dictionary with "messages" key
        conversation = {"messages": []}
        
        # Add messages to the conversation
        for message in chat:
            if isinstance(message, dict) and 'role' in message and 'content' in message:
                conversation["messages"].append({
                    "role": message["role"],
                    "content": message["content"]
                })
        
        # Only add conversations that have at least one message
        if conversation["messages"]:
            conversations.append(conversation)

    # Create train and test datasets
    train_data = conversations # First 2 conversations for training
    test_data = conversations[:100] # Last conversation for testing


    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Handle HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: HDFS support not available. Skipping HDFS copy.")

    # Print statistics
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {local_dir}")


if __name__ == "__main__":
    main()
