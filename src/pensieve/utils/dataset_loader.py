"""Dataset loading utilities for multi-turn conversation evaluation."""

import random
from typing import List, Tuple, Optional


def load_sharept_dataset(
    num_conversations: int = 10,
    max_turns: int = 5,
    min_turns: int = 2,
) -> List[Tuple[str, List[str]]]:
    """Load multi-turn conversations from ConversationChronicles-sharegpt dataset.

    Extracts user messages from conversations and returns them in the format
    compatible with run_concurrent_comparison().

    Note: Dataset contains 3-123 turn conversations. Default max_turns=5 filters
    for shorter conversations; adjust as needed for your benchmark.

    Args:
        num_conversations: Number of conversations to sample
        max_turns: Maximum number of user turns per conversation
        min_turns: Minimum number of user turns per conversation

    Returns:
        List of (session_id, [user_messages]) tuples
        Example: [("session_1", ["Hello", "Tell me about...", ...]), ...]

    Raises:
        ImportError: If datasets library is not installed
        ValueError: If not enough conversations match the criteria
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not found. Install with: pip install datasets"
        )

    print("Loading ShareGPT dataset from HuggingFace...")

    # Load ConversationChronicles-sharegpt dataset (785k multi-turn conversations)
    # Format: "from": "human"/"gpt", "value": message (standard ShareGPT format)
    # Turn range: 3-123 turns per conversation
    dataset = load_dataset(
        "Dans-DiscountModels/ConversationChronicles-sharegpt",
        split="train"
    ) 


    print(f"✓ Dataset loaded: {len(dataset)} conversations available")

    # Filter conversations by turn count
    valid_conversations = []

    for idx, item in enumerate(dataset):
        # Handle both possible dataset formats
        conversations = item.get("conversations", [])

        if not conversations:
            continue

        # Count human messages (user turns)
        human_turns = [c for c in conversations if c.get("from") == "human"]

        # Check if conversation meets turn criteria
        if min_turns <= len(human_turns) <= max_turns:
            # Extract only user messages (model will generate responses)
            user_messages = [c.get("value", "") for c in human_turns]

            if user_messages and all(msg.strip() for msg in user_messages):
                session_id = f"session_{len(valid_conversations) + 1}"
                valid_conversations.append((session_id, user_messages))

                # Stop when we have enough conversations
                if len(valid_conversations) >= num_conversations:
                    break

    if len(valid_conversations) < num_conversations:
        print(
            f"⚠ Warning: Only {len(valid_conversations)} conversations found "
            f"with {min_turns}-{max_turns} turns. Requested {num_conversations}."
        )

    if not valid_conversations:
        raise ValueError(
            f"No conversations found with {min_turns}-{max_turns} turns. "
            "Try adjusting min_turns and max_turns."
        )

    # Sample the requested number of conversations
    sampled = random.sample(valid_conversations, min(num_conversations, len(valid_conversations)))

    # Print statistics
    total_turns = sum(len(msgs) for _, msgs in sampled)
    avg_turns = total_turns / len(sampled) if sampled else 0

    print(f"✓ Filtered to {len(sampled)} conversations")
    print(f"  Average turns per conversation: {avg_turns:.1f}")
    print(f"  Turn distribution: {min_turns}-{max_turns}")

    return sampled
