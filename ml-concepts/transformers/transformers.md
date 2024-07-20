## Understanding Transformers

- **motivation**
    - when we have any piece of text, to the human eye, it’s very easy to understand what’s going on, but more specifically, there are 3 observations that can be made
        1. the encoded input can be surprisingly large — if we use an embedding vector of length 1024, the encoded input would be the number of words x 1024
        2. It is not obvious how to use a FCC in this situation as sentences come in different length
        3. Language is ambiguous — in text, we often use the word “it” to refer to something in the sentence, but we need to get the model to understand what exactly that word is referring to
- **model input**
    - The process begins with tokenizing the input text into individual tokens (words or subwords). Each token is then converted into a vector representation:
        - A lookup table (embedding matrix) is used to convert each token to a dense vector.
        - The dimensionality of this vector is typically 512 or 768 in models like BERT.
        - These embeddings are learned during the training process.
    - aside: **understanding tokenization**
        - there is some preprocessing done on the input, more specifically we tokenize the input text into subwords or words (depending on the tokenizer)
            - for example, BERT uses uses WordPiece, which breaks down words into subwords to handle rare or unknown words efficiently
                
                ```python
                from transformers import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                text = "Tokenizing input is an important step."
                tokens = tokenizer.tokenize(text)
                # Output: ['token', '##izing', 'input', 'is', 'an', 'important', 'step', '.']
                ```
                
        - sometimes models require special tokens — for example the BERT requires `[CLS]` at the beginning and `[SEP]` at the end of the sequence
        - then we convert each token into its corresponding integer ID using the tokenizers vocabulary
            
            ```python
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            # Output: [101, 19204, 2135, 1567, 2003, 2019, 2590, 3350, 1012, 102]
            ```
            
        - then we pad or truncate the sequences to a fixed length if necessary
        - then we create an attention mask which allows the model to differentiate between actual tokens and padding tokens — 1 for real tokens, 0 for padding tokens
            
            ```python
            attention_mask = [1 if id != 0 else 0 for id in padded_token_ids]
            # Output: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
            ```
- **positional encodings**
    
    <aside>
    💡 the purpose of positional encodings is to enable the model to distinguish between identical tokens in different positions — for example, in the sentence “the cat sat on the mat”, the 2 instances of “the” need to be treated differently based on their positions
    
    </aside>
    
    - these positional encodings are really important as they provide information about the position of tokens in the input sequence — remember, transformers are position-agnostic
        - this means they don’t have a built-in notion of the order of the tokens — positional encodings allow the model to leverage the order of the sequence
    - **computing positional encodings**
        - position encodings can be added to the input embeddings using fixed functions — a common approach is to use sine and cosine functions of different frequencies
            - $PE_{(pos, 2i)} = sin(\frac{pos}{10000^{2i/d_{model}}})$
            - $PE_{(pos, 2i+1)} = cos(\frac{pos}{10000^{2i/d_{model}}})$
                - In these equations, *pos* is the position, *i* is the dimension, and $d_{model}$ is the dimension of the model (embedding size)
        - check `positional_encoding.py` for an implementation of this