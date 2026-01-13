"""Test assets for agent testing.

These test fixtures use generic e-commerce customer support examples
to allow testing the pipeline independently of any specific domain.
"""

import json

# =============================================================================
# EXISTING TEST DATA
# =============================================================================

TEST_TRANSCRIPT = """

--- START OF CONVERSATION ---

ASSISTANT: **Welcome! I'm the customer support assistant for TechStore. How can I help you today?**

_Note: Please don't share personal information like passwords or payment details in this chat._

USER: What shipping zones do you deliver to?

ASSISTANT: We deliver to three shipping zones:

- **Zone A (Metro)**: Same-day or next-day delivery available for major metropolitan areas.
- **Zone B (Regional)**: 2-3 business days for regional areas.
- **Zone C (Remote)**: 5-7 business days for remote locations.

Would you like me to check which zone your address falls under? Just share your postal code and I can look it up for you!

--- END OF CONVERSATION ---

"""

TEST_ISSUES_TEXT = """
- Title: How to track order status and delivery updates
  FAQ: How can I track my order status and receive delivery updates for my TechStore purchase?
- Title: Return policy for electronics not working
  FAQ: What is the return policy if my electronic product is defective or not working as expected?
- Title: Website checkout not processing payment
  FAQ: What should I do if the TechStore website is not processing my payment during checkout?
"""

TEST_SINGLE_ISSUE_TEXT = """
{
"issue_title": "Do I qualify for free shipping to Zone B?",
"canonical_faq_question": "I live in a Zone B area, do I qualify for free shipping on my order from TechStore?"
}
"""

# =============================================================================
# TEST DATA FOR ADDITIONAL AGENTS
# =============================================================================

# Executive Report Agent - realistic usage and cluster data
TEST_EXECUTIVE_DATA = {
    "usage": {
        "total_conversations": 1247,
        "total_messages": 4891,
        "date_range_days": 14,
        "avg_conversations_per_day": 89.1,
        "median_messages_per_conversation": 4,
        "first_conversation": "2024-11-01",
        "last_conversation": "2024-11-14",
        "tools": {
            "top_tools_used": {
                "order_lookup": 423,
                "shipping_calculator": 287,
                "product_search": 156,
            },
            "conversations_with_tool_use": 712,
            "conversations_with_citations": 891,
        },
        "message_lengths": {
            "median_user_chars": 45,
            "median_assistant_chars": 312,
        },
        "temporal": {
            "by_hour": {
                "09": 89,
                "10": 134,
                "11": 156,
                "12": 167,
                "13": 145,
                "14": 178,
                "15": 189,
                "16": 145,
                "17": 98,
                "18": 67,
                "19": 45,
                "20": 34,
            },
            "by_weekday": {
                "Monday": 156,
                "Tuesday": 178,
                "Wednesday": 189,
                "Thursday": 234,
                "Friday": 267,
                "Saturday": 145,
                "Sunday": 78,
            },
            "busiest_dates": {
                "2024-11-11": 234,
                "2024-11-12": 312,
                "2024-11-13": 289,
                "2024-11-10": 187,
            },
        },
    },
    "clusters": {
        "method": "kmeans",
        "total_items": 1247,
        "n_clusters": 8,
        "items": [
            {
                "title": "Order Tracking",
                "count": 312,
                "percentage": 25.0,
                "description": "Questions about order status and delivery tracking",
                "detailed_description": "Customers asking about shipment updates, estimated delivery times, and tracking numbers",
            },
            {
                "title": "Returns and Refunds",
                "count": 234,
                "percentage": 18.8,
                "description": "Return policy and refund process inquiries",
                "detailed_description": "Questions about return eligibility, refund timelines, and exchange policies",
            },
            {
                "title": "Shipping Zones",
                "count": 189,
                "percentage": 15.2,
                "description": "Questions about delivery zones and shipping costs",
                "detailed_description": "Address lookups for Zone A, B, and C boundaries and associated shipping fees",
            },
            {
                "title": "Product Information",
                "count": 156,
                "percentage": 12.5,
                "description": "Product specifications and availability",
                "detailed_description": "Questions about product features, stock levels, and compatibility",
            },
            {
                "title": "Website and Technical Issues",
                "count": 123,
                "percentage": 9.9,
                "description": "Problems with the online store",
                "detailed_description": "Login issues, checkout problems, and payment processing errors",
            },
        ],
    },
}

TEST_EXECUTIVE_DATA_JSON = json.dumps(TEST_EXECUTIVE_DATA, indent=2, ensure_ascii=False)

# FAQ Synthesizer Agent - representative question with variants
TEST_FAQ_REPRESENTATIVE = (
    "I live in Zone B, do I qualify for free shipping on orders over $50?"
)

TEST_FAQ_VARIANTS = """- Do Zone B customers get free shipping?
- What is the minimum order for free shipping in regional areas?
- Is Zone B eligible for the free shipping promotion?
- Do I need to spend more for free delivery if I'm in Zone B?"""

# Question Decomposer Agent - compound question for decomposition
TEST_COMPOUND_QUESTION = "What are the store locations near downtown and where can I find information about holiday hours?"

# Simple question that should not be decomposed
TEST_SIMPLE_QUESTION = "What are your store hours on weekends?"

# Question with potential presuppositions (for with_presuppositions mode)
TEST_PRESUPPOSITION_QUESTION = "Can I get same-day delivery to my home in Zone B?"
