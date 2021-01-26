# Formats for data in `.pkl.gz` files

### `tweets_*.pkl.gz` 
pandas DataFrame
   
### `words_per_topic.pkl.gz` 
dict formatted as:

```python
words_per_topic = {
  0: [
    {
      "text": "word",
      "value": 13
    },
    {
      "text": "next word",
      "value": 15
    }
  ],
  1: [
        {
      "text": "word",
      "value": 9
    },
    {
      "text": "next word",
      "value": 23
    }
  ]
}
```

### `words_counts.pkl.gz` 
dict formatted as:

```python
topics_distributions = {
    'per_user': {
        'username': [
            {
                "text": "word",
                 "value": 13
             },
            {
                "text": "next word",
                "value": 15
             }
        ]
    },
    'per_party': {
        'party': [
            {
                "text": "word",
                 "value": 13
             },
            {
                "text": "next word",
                "value": 15
             }
        ]
    },
    'per_coalition': {
        'coalition': [
            {
                "text": "word",
                 "value": 13
             },
            {
                "text": "next word",
                "value": 15
             }
        ]
    }
}
```

### `topics_distributions.pkl.gz` 
dict formatted as:

```python
words_counts = {
    'per_user': {
        'username': [
            {
                'topic': 0,
                'part': 0.02
            },
            {
                'topic': 1,
                'part': 0.12
            }
        ]
    },
    'per_party': {
        'party': [
            {
                'topic': 0,
                'part': 0.02
            },
            {
                'topic': 1,
                'part': 0.12
            }
        ]
    },
    'per_coalition': {
        'coalition': [
            {
                'topic': 0,
                'part': 0.02
            },
            {
                'topic': 1,
                'part': 0.12
            }
        ]
    }
}
```

### `sentiment_distributions.pkl.gz`
dict formatted as:

```python
sentiment_distributions = {
    'per_user': {
        'username': [('positive', 0.2), ('negative', 0.4)]
    },
    'per_party': {},
    'per_coalition': {},
    'per_topic': {}
}
```