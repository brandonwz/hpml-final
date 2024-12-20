DEVICE="cuda"
OUTPUT_OFFSET = 2
INSTRUCT_OFFSET = 4
SAFE_STR = "safe"
UNSAFE_STR = "unsafe"
DBG = False
MAX_EVAL_ITERATIONS = 100
CATEGORIES = {
                "S1":"sexual",
                "S2":"sexual/minors",
                "S3":"harassment",
                "S4":"self-harm",
                "S5":"violence",
                "S6":"hate",
                "S7":"violence/graphic",
                "S8":"self-harm/intent",
                "S9":"harassment/threatening",
                "S10":"self-harm/instructions",
                "S11":"hate/threatening"
            }
EOT_ID = "<|eot_id|>"

# May need to be set lower based off system memory
PROFILE_EVAL_LEN = 100