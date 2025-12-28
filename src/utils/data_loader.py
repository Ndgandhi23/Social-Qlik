"""Data loading utilities and sample dataset."""

from typing import List, Tuple

import numpy as np


def load_sample_data() -> Tuple[List[str], List[str]]:
    """Load the sample interest dataset.
    
    Returns:
        Tuple of (texts, labels) where:
            - texts: List of activity descriptions
            - labels: List of category labels
    """
    data = {
        "Sports / Fitness": [
            "join the weekend volleyball team practice",
            "basketball open court sunday afternoon",
            "soccer 5v5 friendly match this saturday",
            "morning jogging group for beginners",
            "climbing gym training partners needed",
            "football viewing party + pickup game",
            "evening yoga class outdoor park",
            "cycling club long ride sunday morning",
            "tennis practice partners weekdays evening",
            "gym strength training workout buddies",
            "swimming practice group saturday mornings",
            "softball team looking for new players",
            "basketball tournament prep team",
            "crossfit beginner fitness group",
            "hiking trip planning for weekend",
            "rowing team practice sessions open",
            "marathon training running crew",
            "pickleball weekend games meetup",
            "triathlon training partners wanted",
            "boxing fitness training evenings",
            "table tennis casual games group",
            "badminton doubles practice weekday nights",
            "outdoor bootcamp fitness class",
            "weightlifting technique practice crew",
            "rollerblading fun rides sunday park",
            "beach volleyball summer sessions",
            "snowboarding weekend trip planning",
            "ski trip winter sports meetup",
            "kayaking river adventure group",
            "rock climbing bouldering practice night",
            "football fantasy league discussion group",
            "basketball shooting drills practice",
            "track sprint training evenings",
            "soccer penalty kick practice fun",
            "golf swing practice weekend mornings",
            "ultimate frisbee casual matches friday",
            "cheerleading dance practice group",
            "lacrosse beginners practice club",
            "rugby casual sunday league games",
            "skateboarding tricks practice saturday",
            "surfing lessons summer beach group",
            "mountain biking downhill practice team",
            "archery beginner lessons weekday evenings",
            "cricket weekend friendly matches",
            "fencing practice nights sports hall",
            "rowing erg training weekday mornings",
            "open water swim training club",
            "ice hockey casual winter league",
            "handball indoor practice crew"
        ],
        "Academic / Study": [
            "calculus homework help study session",
            "physics exam review group monday night",
            "chemistry lab prep discussion team",
            "statistics problem solving study circle",
            "linear algebra theory reading group",
            "computer science project collab partners",
            "machine learning paper reading club",
            "data science coding practice sessions",
            "real analysis theorems discussion group",
            "economics case study review crew",
            "biology weekly quiz prep team",
            "philosophy text analysis study partners",
            "literature classics reading group tuesdays",
            "psychology research methods review team",
            "history midterm revision discussion",
            "political science debate practice group",
            "engineering design project collaboration",
            "environmental science field study planning",
            "statistics R programming practice partners",
            "algebra proofs challenge problem set",
            "geometry olympiad prep study club",
            "french language grammar practice circle",
            "spanish conversation practice group",
            "german vocabulary drill partners",
            "philosophy logic problem discussion crew",
            "abstract algebra graduate reading team",
            "numerical analysis coding lab help",
            "database systems exam prep group",
            "operating systems weekly quiz review",
            "computer networks assignment help forum",
            "art history text discussion seminar",
            "classical music theory study partners",
            "linguistics syntax paper review club",
            "ethics and law case study analysis",
            "international relations policy paper group",
            "finance quantitative analysis study crew",
            "accounting principles homework group",
            "marketing case competition prep team",
            "management strategy textbook discussion",
            "entrepreneurship business plan workshop",
            "sociology theory reading discussion",
            "archaeology artifact research planning",
            "astronomy observation lab prep crew",
            "robotics design homework collab group",
            "ai ethics policy paper discussion",
            "neuroscience weekly research article club",
            "geology field trip preparation team",
            "oceanography final exam prep partners",
            "meteorology weather model study crew"
        ],
        "Hobbies / Creative": [
            "photography sunset shoot meetup",
            "digital art illustration tips workshop",
            "ceramics pottery wheel practice group",
            "creative writing prompts weekly circle",
            "poetry reading and feedback club",
            "book club mystery novel discussion",
            "board game night strategy sessions",
            "tabletop rpg campaign planning group",
            "painting landscapes weekend workshop",
            "drawing portrait techniques practice",
            "cooking international recipes exchange",
            "baking bread beginners kitchen group",
            "knitting crafts weekly meetup",
            "crochet pattern sharing hobby club",
            "gardening plant care weekend group",
            "urban farming tips exchange partners",
            "bird watching sunday park walks",
            "stargazing astronomy hobby nights",
            "meteor shower viewing event crew",
            "film photography darkroom practice",
            "videography short film collab team",
            "music jamming acoustic guitar nights",
            "band practice rock music weekend",
            "choir singing rehearsals weekday evenings",
            "orchestra classical music practice club",
            "karaoke fun singing friday nights",
            "dance choreography hip hop practice",
            "ballet dance beginner lessons group",
            "salsa dance social practice sessions",
            "latin dance performance rehearsal team",
            "improv comedy acting practice group",
            "theater play script reading rehearsals",
            "lego creative builds weekend challenges",
            "robotics hobby maker space collab",
            "electronics diy circuit building crew",
            "3d printing design modeling practice",
            "woodworking furniture building weekend",
            "calligraphy handwriting art class",
            "origami folding design workshop",
            "cosplay costume making sewing group",
            "fashion design sketching weekend studio",
            "jewelry making beadwork hobby nights",
            "soap making natural crafts club",
            "candle making diy craft evenings",
            "resin art coaster design workshop",
            "scrapbooking photo album creative crew",
            "graphic design poster challenge sessions",
            "animation storyboarding collab team"
        ],
        "Social / Interest": [
            "coffee tasting meetup downtown saturday",
            "brunch social sunday morning gathering",
            "networking event tech professionals friday",
            "language exchange spanish english partners",
            "french conversation practice social club",
            "game night friends board games meetup",
            "pub trivia quiz team weeknight fun",
            "movie night film lovers friday crew",
            "comedy show group outing saturday night",
            "karaoke singing social evening downtown",
            "picnic afternoon park casual hangout",
            "city walking tour historical landmarks",
            "museum visit art lovers sunday group",
            "theater play group tickets discount offer",
            "concert live music outing friends crew",
            "sports bar watch party basketball finals",
            "camping weekend outdoor adventure team",
            "hiking day trip social fitness group",
            "beach bonfire evening hangout friday",
            "road trip planning summer travel crew",
            "restaurant foodies dinner night out",
            "cooking class date night couples event",
            "wine tasting vineyard weekend trip",
            "craft beer brewery tour tasting session",
            "cocktail mixing workshop friends night",
            "holiday party planning friends committee",
            "birthday surprise group coordination team",
            "volunteering animal shelter weekend help",
            "charity run fundraising planning team",
            "community garden planting volunteer crew"
        ]
    }
    
    texts = []
    labels = []
    
    for category, items in data.items():
        texts.extend(items)
        labels.extend([category] * len(items))
    
    return texts, labels


def create_balanced_split(
    texts: List[str],
    labels: List[str],
    train_ratio: float = 0.8,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Create stratified train/test split.
    
    Args:
        texts: Text descriptions
        labels: Category labels
        train_ratio: Fraction for training
        random_state: Random seed
        
    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels)
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(
        texts,
        labels,
        train_size=train_ratio,
        random_state=random_state,
        stratify=labels
    )