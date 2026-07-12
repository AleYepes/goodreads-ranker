import asyncio
import httpx

API_URL = "https://kxbwmqov6jgg3daaamb744ycu4.appsync-api.us-east-1.amazonaws.com/graphql" # Apparently stable
HEADERS = {
    "content-type": "application/json",
    "x-api-key": "da2-xpgsdydkbregjhpr6ejzqdhuwy", # Apparently stable
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

async def fetch_complete_book_data(legacy_book_id, similar_limit=20):
    book_data = None
    sim_data = {}
    social_data = []
    async with httpx.AsyncClient() as client:
        try:
            details_payload = {
                "operationName": "getBookByLegacyId",
                "variables": {"legacyBookId": int(legacy_book_id)},
                "query": """
                query getBookByLegacyId($legacyBookId: Int!) {
                  getBookByLegacyId(legacyId: $legacyBookId) {
                    __typename
                    id
                    legacyId
                    title
                    titleComplete
                    webUrl
                    descriptionStripped: description(stripped: true)
                    primaryContributorEdge {
                      node {
                        id
                        legacyId
                        name
                        description
                        isGrAuthor
                        works {
                          totalCount
                        }
                        webUrl
                        followers {
                          totalCount
                        }
                      }
                      role
                    }
                    secondaryContributorEdges {
                      node {
                        id
                        legacyId
                        name
                        description
                        isGrAuthor
                        works {
                          totalCount
                        }
                        webUrl
                        followers {
                          totalCount
                        }
                      }
                      role
                    }
                    bookSeries {
                      userPosition
                      series {
                        id
                        title
                        webUrl
                      }
                    }
                    bookGenres {
                      genre {
                        name
                        webUrl
                      }
                    }
                    details {
                      asin
                      format
                      numPages
                      publicationTime
                      publisher
                      isbn
                      isbn13
                      language {
                        name
                      }
                    }
                    work {
                      id
                      legacyId
                      stats {
                        averageRating
                        ratingsCount
                        ratingsCountDist
                        textReviewsCount
                        textReviewsLanguageCounts {
                          count
                          isoLanguageCode
                        }
                      }
                      details {
                        webUrl
                        shelvesUrl
                        publicationTime
                        originalTitle
                        places {
                          name
                          countryName
                          webUrl
                          year
                        }
                        characters {
                          name
                          webUrl
                        }
                        awardsWon {
                          name
                          webUrl
                          awardedAt
                          category
                          designation
                        }
                      }
                      bestBook {
                        id
                        legacyId
                      }
                      editions {
                        webUrl
                      }
                    }
                  }
                }
                """
            }
            response = await client.post(API_URL, json=details_payload, headers=HEADERS, timeout=10.0)
            book_data = response.json()

            # book_node = book_data.get("data", {}).get("getBookByLegacyId")
            # kca_id = book_node.get("id")
            # if kca_id:
            #     similar_payload = {
            #         "operationName": "GetSimilarBooks",
            #         "variables": {
            #             "id": kca_id,
            #             "limit": similar_limit
            #         },
            #         "query": """
            #         query GetSimilarBooks($id: ID!, $limit: Int!) {
            #           getSimilarBooks(id: $id, pagination: {limit: $limit}) {
            #             edges {
            #               node {
            #                 title
            #                 webUrl
            #                 work {
            #                   stats {
            #                     averageRating
            #                     ratingsCount
            #                   }
            #                 }
            #               }
            #             }
            #           }
            #         }
            #         """
            #     }
            #     sim_response = await client.post(API_URL, json=similar_payload, headers=HEADERS, timeout=10.0)
            #     sim_data = sim_response.json()

            #     social_payload = {
            #         "operationName": "GetSocialSignals",
            #         "variables": {"bookId": kca_id},
            #         "query": """
            #         query GetSocialSignals($bookId: ID!) {
            #           getSocialSignals(bookId: $bookId, shelfStatus: [CURRENTLY_READING, TO_READ]) {
            #             name
            #             count
            #             shelfPhrase
            #             userPhrase
            #           }
            #         }
            #         """
            #     }
            #     social_response = await client.post(API_URL, json=social_payload, headers=HEADERS, timeout=10.0)
            #     social_data = social_response.json().get("data", {}).get("getSocialSignals") or []

            # return book_data, sim_data, social_data
            return book_data

        except Exception as e:
            print(f"Error fetching data for legacy book {legacy_book_id}: {e}")
            return None

async def main():
    test_ids = [1]
    for book_id in test_ids:
        # book_data, sim_data, social_data = await fetch_complete_book_data(book_id)
        book_data = await fetch_complete_book_data(book_id)
        print(f'\nBook data:\n{book_data}')
        # print(f'\nSimilarity data:\n{sim_data}')
        # print(f'\nSocial data:\n{social_data}')

if __name__ == "__main__":
    asyncio.run(main())