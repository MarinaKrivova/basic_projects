class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting

        
    def count_docs_in_collection(self, index):
        """Counts number of unique documents in the index file"""
        doc_in_collection = set([])
        for main_key in self.index:
            doc_in_collection.update(self.index[main_key].keys())
        return len(doc_in_collection)
    
    def cosine_similarity_final(self, dict_qi_di, dict_len_doc):
        """Computes cosine similarities of documnets vectors and returns top10 most similar documents"""
        # similarity ~ sum(q_i*d_i)/sum(d_i^2) = sum(q_i*d_i)/len_doc
        import math
        
        doc_sim_dict = {}
        for doc_index in dict_qi_di:    
            doc_sim_dict[doc_index] = dict_qi_di[doc_index] / math.sqrt(dict_len_doc[doc_index])
        
        sorted_docs = sorted(doc_sim_dict, key = lambda w: doc_sim_dict[w], reverse=True)
        if len(sorted_docs)>=10:
            return sorted_docs[:10]
        else:
            return sorted_docs    

    def BinaryModel(self, query):
        """Computes similarity of a query and documents based on binary term weighting
        and return indexes of top 10 relevant documents"""

        # similarity ~ sum(q_i*d_i)/sum(d_i^2) = sum(q_i*d_i)/len_doc
        # q_i*d_i = terms 0 or 1 for query and a document
        
        # retreive unique documents that have at least one word from the query
        # and immediatly compute product of word frequencies in query and this document 

        doc_index_qi_di = {}
        words_in_query_and_index  = list(query & self.index.keys())
        for word_query in words_in_query_and_index:
            for doc_index in self.index[word_query]:
                doc_index_qi_di[doc_index] = doc_index_qi_di.setdefault(doc_index,0) + 1 * 1
        
        # The retrieved documents have obviously more words than presented in the query,
        # so to access full length of a document (sum(d_i^2),
        # we need to go through the whole index files and find other words in each retrieved document

        doc_len_dict = {}
        for word_general in self.index:
            key_doc_in_index_word_general = list(doc_index_qi_di.keys() & self.index[word_general].keys())
            for key_doc in key_doc_in_index_word_general:
                doc_len_dict[key_doc] = doc_len_dict.setdefault(key_doc, 0) + 1**2

        retrieved_docs = self.cosine_similarity_final(doc_index_qi_di, doc_len_dict)
        return retrieved_docs


    def tfModel(self, query):
        """Computes similarity of a query and documents based on frequency term weighting
        and return indexes of top 10 relevant documents"""
        
        # similarity ~ sum(q_i*d_i)/sum(d_i^2) = sum(q_i*d_i)/len_doc
        # q_i*d_i = tf (frequencies) for query and a document

        doc_index_tfqi_tfdi = {}
        words_in_query_and_index  = list(query & self.index.keys())
        for word_query in words_in_query_and_index:
            for doc_index in self.index[word_query]:
                doc_index_tfqi_tfdi[doc_index] = doc_index_tfqi_tfdi.setdefault(doc_index, 0) + query[word_query] * self.index[word_query][doc_index]
        
        doc_len_dict = {}
        for word_general in self.index:
            key_doc_in_index_word_general = list(doc_index_tfqi_tfdi.keys() & self.index[word_general].keys())
            for key_doc in key_doc_in_index_word_general:
                doc_len_dict[key_doc] = doc_len_dict.setdefault(key_doc,0) + self.index[word_general][key_doc]**2
                         
        retrieved_docs = self.cosine_similarity_final(doc_index_tfqi_tfdi, doc_len_dict)
        return retrieved_docs

    def tfidfModel(self, query):
        """Computes similarity of a query and documents based on frequency term weighting
        and return indexes of top 10 relevant documents"""
        
        # similarity ~ sum(q_i*d_i)/sum(d_i^2) = sum(q_i*d_i)/len_doc
        # q_i*d_i = tfidf for query and a document
        
        
        # df - number of doc containg term
        # tf - number of times term in the doc
        # idf = math.log10(D/df)
        # D - number of documents in the collection
        
        import math
        D = self.count_docs_in_collection(self.index)
        
        tfidf_doc_query = {}
        query_in_index = list(query.keys() & self.index.keys())
        for word_query in query_in_index:
            df = len(self.index[word_query])
                # idf is identical for query and document
            idf = math.log10(D/df)
            for doc_index in self.index[word_query]:
                tf_doc = self.index[word_query][doc_index]
                tf_query = query[word_query]
                tfidf_doc_query[doc_index] = tfidf_doc_query.setdefault(doc_index, 0) + tf_query*idf*tf_doc*idf

        doc_tfidf_len = {}
        for word_general in self.index:
            df_general = len(self.index[word_general])
            idf_general = math.log10(D/df_general)
            key_word_query_index_word_doc = list(tfidf_doc_query.keys() & self.index[word_general].keys())
            for key_doc in key_word_query_index_word_doc:
                tf_doc_general = self.index[word_general][key_doc]
                doc_tfidf_len[key_doc] = doc_tfidf_len.setdefault(key_doc, 0) + (tf_doc_general*idf_general)**2 
        
        retrieved_docs = self.cosine_similarity_final(tfidf_doc_query, doc_tfidf_len)
        return retrieved_docs
    
    def forQuery(self, query):
        if self.termWeighting == 'binary':
            return self.BinaryModel(query)
        elif self.termWeighting == 'tf':
            return self.tfModel(query)
        elif self.termWeighting == 'tfidf':
            return self.tfidfModel(query)
        else:
            print("ERROR: term weighting schemes is not recognized")
            return