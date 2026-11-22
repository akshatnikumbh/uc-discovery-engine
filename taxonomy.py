"""Shared labels for the Urban Company repeat-booking discovery engine.

Every label the LLM may return lives here, so collection, classification and the
dashboard agree. Edit this file, not the prompts, when the taxonomy changes.
"""

CATEGORIES = [
    "cleaning", "pest_control", "ac_appliance", "electrician_plumber_carpenter",
    "salon_women", "salon_men", "spa_massage", "painting_revamp", "water_purifier_lock",
    "instahelp", "other", "unclear",
]

# What kind of post this is. need_post matters most for the silent-churner problem:
# someone asking "where do I find a plumber" is a need UC could have served.
RECORD_TYPES = [
    "service_experience",   # describes a booking that was done
    "need_post",            # asks how or where to get a home service done
    "comparison",           # UC versus local options or other apps
    "general_opinion",      # views about UC or home services, no specific event
    "irrelevant",
]

RETURN_SIGNALS = [
    "returned_same_category", "returned_other_category", "regular_user",
    "did_not_return", "switched_to_alternative", "plans_to_return", "none", "unclear",
]

# Why a customer did not come back, or chose something else, in their own account.
NOT_RETURN_REASONS = [
    "no_need_arose", "long_service_cycle", "did_not_think_of_uc", "unaware_uc_offers_it",
    "local_provider_faster", "small_job_not_worth_app", "price_or_visit_charge",
    "wanted_same_professional", "went_direct_to_professional", "domestic_help_handles",
    "society_vendor_or_rwa", "slot_or_availability", "bad_experience_quality",
    "bad_experience_behaviour", "refund_or_support_issue", "trust_safety_in_home",
    "urgent_need", "diy", "other", "none",
]

NEED_TRIGGERS = [
    "breakdown_or_fault", "seasonal", "festival_or_event", "guests", "move_in_or_out",
    "routine_schedule", "noticed_dirt_or_pest", "self_care_routine", "recommendation", "none",
]

ALTERNATIVES = [
    "local_provider", "society_whatsapp_or_rwa", "neighbour_or_watchman_referral",
    "domestic_help", "direct_with_same_pro", "other_app_or_justdial",
    "brand_service_center", "diy", "postponed_or_skipped", "none",
]

POST_SERVICE_EXPECTATIONS = [
    "reminder_when_due", "care_tips", "warranty_clarity", "rebook_same_pro",
    "follow_up_check", "nothing_expected", "none",
]

ACQUISITION = [
    "referral", "search_when_urgent", "ad_or_social", "society_or_onsite",
    "already_had_app", "unknown",
]

# Addressability: how far a customer-side product change that offers no discount, no
# membership, no price change and no change to how professionals are managed could act
# on this reason (0 to 1). Judgement weights: record the reasoning in the Evidence Log.
ADDRESSABILITY = {
    "did_not_think_of_uc": 1.0,          # recall at the moment a need appears
    "unaware_uc_offers_it": 1.0,         # breadth awareness
    "long_service_cycle": 0.6,           # timing help, but risk of nudging unneeded services
    "small_job_not_worth_app": 0.4,
    "local_provider_faster": 0.4,
    "trust_safety_in_home": 0.5,
    "wanted_same_professional": 0.5,     # customer-side rebooking, not pro management
    "urgent_need": 0.4,
    "society_vendor_or_rwa": 0.4,
    "went_direct_to_professional": 0.3,
    "diy": 0.3,
    "no_need_arose": 0.2,                # creating need risks pushing unneeded services
    "bad_experience_quality": 0.2,       # brief: do not frame as bad experience
    "bad_experience_behaviour": 0.2,
    "refund_or_support_issue": 0.2,
    "slot_or_availability": 0.2,         # supply side
    "domestic_help_handles": 0.1,
    "other": 0.2,
    "price_or_visit_charge": 0.0,        # pricing is out of bounds
    "none": 0.0,
}

NON_RETURN_SIGNALS = {"did_not_return", "switched_to_alternative"}
