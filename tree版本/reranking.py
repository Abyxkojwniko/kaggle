#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def delete(dataset):
    for topic_id in topics_df.query("has_content").sample(n=1000).index:
        topic = Topic(topic_id)
        if any(topic.language != content.language for content in topic.content):
            topic.content.remove(content)
            nonmatching += 1
        else:
            matching += 1

    print("Matching:", matching)
    print("Nonmatching:", nonmatching)
    print("Percent matching: {:.2f}%".format(100 * matching / (matching + nonmatching)))

