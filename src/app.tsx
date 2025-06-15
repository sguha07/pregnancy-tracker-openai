import React, { useState, useEffect, useRef } from 'react';
import { Calendar, Pill, AlertCircle, Apple, MessageCircle, Home, Baby, ChevronRight, Send, X, Search, Heart, Activity, FileText, AlertTriangle, CheckCircle } from 'lucide-react';
import OpenAI from 'openai';

// ==================== TYPES ====================
type Nutrient = {
  nutrient: string;
  amount: string;
  unit: string;
  category: string;
};

type WeightGainRecommendation = {
  prePregnancyBMI: string;
  bmiRange: string;
  recommendedGain: string;
  unit: string;
};

type Symptom = {
  sign: string;
  urgency: string;
  action: string;
  severity: 'low' | 'medium' | 'high';
  category?: string;
};

type Medication = {
  drug: string;
  brand?: string;
  safety: string;
  safetyLevel: string;
  note?: string;
  condition?: string;
};

type WeekInfo = {
  trimester: string;
  title: string;
  commonSymptoms: Array<{ symptom: string; status: string }>;
  exercise?: {
    name: string;
    benefits: string;
    instructions: string[];
  };
};

type Message = {
  role: 'user' | 'assistant';
  content: string;
  source?: 'knowledge-base' | 'ai-general' | 'error';
};

type Tab = 'home' | 'tracker' | 'medications' | 'symptoms' | 'nutrition' | 'emergency' | 'chat';

type KnowledgeSection = {
  id: string;
  content: string;
  embedding?: number[];
};

// ==================== KNOWLEDGE BASE SERVICE ====================
class KnowledgeBaseService {
  private knowledgeBase: any = null;
  private sections: KnowledgeSection[] = [];
  private openai: OpenAI | null = null;
  private embeddingsGenerated = false;

  constructor() {
    const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
    if (apiKey) {
      this.openai = new OpenAI({
        apiKey: apiKey,
        dangerouslyAllowBrowser: true // Note: In production, use a backend proxy
      });
    }
  }

  async initialize() {
    // Load knowledge base
    try {
      const response = await fetch('/knowledgeBase.json');
      const data = await response.json();
      this.knowledgeBase = data.pregnancyKnowledgeGraph;
      
      // Convert knowledge base to searchable sections
      this.createSections();
      
      // Generate embeddings if OpenAI is available
      if (this.openai) {
        await this.generateEmbeddings();
      }
    } catch (error) {
      console.error('Failed to load knowledge base:', error);
    }
  }

  private createSections() {
    // Flatten knowledge base into searchable sections
    const kb = this.knowledgeBase;
    
    // Nutritional requirements
    this.sections.push({
      id: 'nutrition-daily',
      content: `Daily nutritional requirements during pregnancy: ${kb.nutritionalRequirements.dailyMacros.map((n: Nutrient) => 
        `${n.nutrient}: ${n.amount} ${n.unit} (${n.category})`).join(', ')}`
    });

    // Weight gain recommendations
    this.sections.push({
      id: 'nutrition-weight',
      content: `Weight gain recommendations: ${kb.nutritionalRequirements.weightGainRecommendations.map((w: WeightGainRecommendation) => 
        `${w.prePregnancyBMI} (BMI ${w.bmiRange}): ${w.recommendedGain} ${w.unit}`).join(', ')}`
    });

    // Food safety
    this.sections.push({
      id: 'food-safety',
      content: `Foods to avoid during pregnancy: Unsafe seafood (${kb.foodSafety.seafoodGuidelines.unsafe.join(', ')}), 
        ${kb.foodSafety.avoidFoods.map((f: any) => f.item).join(', ')}`
    });

    // Morning sickness
    this.sections.push({
      id: 'morning-sickness',
      content: `Morning sickness management: Eat ${kb.morningSicknessManagement.whatToEat.join(', ')}. 
        Avoid ${kb.morningSicknessManagement.avoidFoods.join(', ')}. 
        Tips: ${kb.morningSicknessManagement.eatingTips.join(', ')}`
    });

    // Pregnancy timeline
    Object.entries(kb.pregnancyTimeline).forEach(([key, value]: [string, any]) => {
      this.sections.push({
        id: `timeline-${key}`,
        content: `${value.title} (${value.trimester} trimester): Common symptoms include ${
          value.commonSymptoms.map((s: any) => `${s.symptom} - ${s.status}`).join(', ')
        }. Recommended exercise: ${value.exercise?.name} - ${value.exercise?.benefits}`
      });
    });

    // Symptoms
    kb.symptomTroubleshooting.categories.forEach((cat: any) => {
      cat.symptoms.forEach((symptom: any) => {
        this.sections.push({
          id: `symptom-${symptom.sign.toLowerCase().replace(/\s+/g, '-')}`,
          content: `${symptom.sign} (${cat.category}): ${symptom.action}. Urgency: ${symptom.urgency}. Severity: ${symptom.severity}`
        });
      });
    });

    // Medications
    kb.medications.byCondition.forEach((condition: any) => {
      condition.medications.forEach((med: any) => {
        this.sections.push({
          id: `medication-${med.drug.toLowerCase()}`,
          content: `${med.drug} (${med.brand || 'Generic'}) for ${condition.condition}: ${med.safetyLevel}. ${med.note || ''}`
        });
      });
    });
  }

  private async generateEmbeddings() {
    if (!this.openai || this.embeddingsGenerated) return;

    try {
      // Generate embeddings for all sections
      const embeddingPromises = this.sections.map(async (section) => {
        const response = await this.openai!.embeddings.create({
          model: 'text-embedding-ada-002',
          input: section.content,
        });
        section.embedding = response.data[0].embedding;
      });

      await Promise.all(embeddingPromises);
      this.embeddingsGenerated = true;
    } catch (error) {
      console.error('Failed to generate embeddings:', error);
    }
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }

  async findRelevantSections(query: string, topK: number = 3): Promise<KnowledgeSection[]> {
    if (!this.openai || !this.embeddingsGenerated) {
      // Fallback to simple keyword search
      const queryLower = query.toLowerCase();
      return this.sections
        .filter(section => section.content.toLowerCase().includes(queryLower))
        .slice(0, topK);
    }

    try {
      // Generate embedding for the query
      const response = await this.openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: query,
      });
      const queryEmbedding = response.data[0].embedding;

      // Calculate similarities
      const sectionsWithScores = this.sections
        .filter(section => section.embedding)
        .map(section => ({
          section,
          score: this.cosineSimilarity(queryEmbedding, section.embedding!)
        }))
        .sort((a, b) => b.score - a.score);

      // Return top K sections with score > 0.7
      return sectionsWithScores
        .filter(item => item.score > 0.7)
        .slice(0, topK)
        .map(item => item.section);
    } catch (error) {
      console.error('Failed to find relevant sections:', error);
      return [];
    }
  }

  getKnowledgeBase() {
    return this.knowledgeBase;
  }

  getWeekInfo(week: number): WeekInfo | null {
    if (!this.knowledgeBase) return null;
    if (week >= 9 && week <= 12) return this.knowledgeBase.pregnancyTimeline.weeks9to12;
    if (week >= 13 && week <= 16) return this.knowledgeBase.pregnancyTimeline.weeks13to16;
    return null;
  }

  checkMedicationSafety(medName: string): Medication[] {
    if (!this.knowledgeBase) return [];
    const results: Medication[] = [];
    this.knowledgeBase.medications.byCondition.forEach((condition: any) => {
      condition.medications.forEach((med: any) => {
        if (med.drug.toLowerCase().includes(medName.toLowerCase()) ||
            (med.brand && med.brand.toLowerCase().includes(medName.toLowerCase()))) {
          results.push({ ...med, condition: condition.condition });
        }
      });
    });
    return results;
  }

  getSymptomInfo(symptom: string): Symptom[] {
    if (!this.knowledgeBase) return [];
    const results: Symptom[] = [];
    this.knowledgeBase.symptomTroubleshooting.categories.forEach((cat: any) => {
      cat.symptoms.forEach((s: any) => {
        if (s.sign.toLowerCase().includes(symptom.toLowerCase())) {
          results.push({ ...s, category: cat.category });
        }
      });
    });
    return results;
  }

  getEmergencySymptoms(): Symptom[] {
    if (!this.knowledgeBase) return [];
    const emergencies: Symptom[] = [];
    this.knowledgeBase.symptomTroubleshooting.categories.forEach((cat: any) => {
      cat.symptoms.forEach((s: any) => {
        if (s.severity === 'high') {
          emergencies.push({ ...s, category: cat.category });
        }
      });
    });
    return emergencies;
  }

  getNutritionalRequirements() {
    if (!this.knowledgeBase) return { dailyMacros: [], weightGainRecommendations: [] };
    return this.knowledgeBase.nutritionalRequirements;
  }
}

// ==================== MAIN APP COMPONENT ====================
const PregnancyTrackerApp: React.FC = () => {
  const [kb] = useState(() => new KnowledgeBaseService());
  const [isKbLoaded, setIsKbLoaded] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>('home');
  const [dueDate, setDueDate] = useState<string>(() => {
    const saved = localStorage.getItem('pregnancyDueDate');
    return saved ? saved : '';
  });
  const [modalVersion, setModalVersion] = useState<number>(() => {
    const saved = localStorage.getItem('pregnancyDueDate');
    return (!saved || saved === '') ? 1 : 0;
  });
  const showDueDateModal = modalVersion > 0;
  const [currentWeek, setCurrentWeek] = useState<number>(() => {
    const saved = localStorage.getItem('pregnancyDueDate');
    if (!saved) return 12;
    const today = new Date();
    const due = new Date(saved);
    const diffTime = due.getTime() - today.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    const weeksRemaining = Math.floor(diffDays / 7);
    return Math.max(1, Math.min(40, 40 - weeksRemaining));
  });
  const [medicationSearch, setMedicationSearch] = useState<string>('');
  const [symptomSearch, setSymptomSearch] = useState<string>('');
  const [chatMessages, setChatMessages] = useState<Message[]>([
    { 
      role: 'assistant', 
      content: 'Hello! I\'m your pregnancy assistant powered by OpenAI. Ask me about symptoms, medications, nutrition, or anything pregnancy-related!',
      source: 'knowledge-base'
    }
  ]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initialize knowledge base
    kb.initialize().then(() => {
      setIsKbLoaded(true);
    });
  }, [kb]);

  useEffect(() => {
    console.log('Modal version changed:', modalVersion, 'showModal:', showDueDateModal);
  }, [modalVersion, showDueDateModal]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  useEffect(() => {
    if (dueDate) {
      const today = new Date();
      const due = new Date(dueDate);
      const diffTime = due.getTime() - today.getTime();
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
      const weeksRemaining = Math.floor(diffDays / 7);
      const calculatedWeek = Math.max(1, Math.min(40, 40 - weeksRemaining));
      setCurrentWeek(calculatedWeek);
    }
  }, [dueDate]);

  const handleDueDateSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const inputDate = (form.elements.namedItem('dueDate') as HTMLInputElement).value;
    if (inputDate) {
      setDueDate(inputDate);
      localStorage.setItem('pregnancyDueDate', inputDate);
      setModalVersion(0);
    }
  };

  const weekInfo = kb.getWeekInfo(currentWeek);
  const emergencySymptoms = kb.getEmergencySymptoms();
  const nutritionalReqs = kb.getNutritionalRequirements();

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || isLoading) return;
    
    const userMessage: Message = { role: 'user', content: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setIsLoading(true);
    
    try {
      // Find relevant knowledge base sections
      const relevantSections = await kb.findRelevantSections(chatInput, 3);
      
      let assistantMessage: Message;
      
      if (!import.meta.env.VITE_OPENAI_API_KEY) {
        // Fallback if no API key
        assistantMessage = {
          role: 'assistant',
          content: "OpenAI API key not configured. Please set VITE_OPENAI_API_KEY in your environment variables.",
          source: 'error'
        };
      } else {
        // Prepare context from knowledge base
        const kbContext = relevantSections.length > 0 
          ? `Based on our pregnancy knowledge base:\n${relevantSections.map(s => s.content).join('\n\n')}`
          : '';

        // Call OpenAI
        const openai = new OpenAI({
          apiKey: import.meta.env.VITE_OPENAI_API_KEY,
          dangerouslyAllowBrowser: true
        });

        const completion = await openai.chat.completions.create({
          model: 'gpt-3.5-turbo',
          messages: [
            {
              role: 'system',
              content: `You are a helpful pregnancy care assistant. You have access to a medical knowledge base about pregnancy. 
                When answering questions, clearly indicate whether your response is based on the provided knowledge base or general knowledge.
                Always recommend consulting healthcare providers for medical decisions.
                ${kbContext ? `\n\nRelevant information from knowledge base:\n${kbContext}` : ''}`
            },
            {
              role: 'user',
              content: chatInput
            }
          ],
          temperature: 0.7,
          max_tokens: 500
        });

        const responseContent = completion.choices[0].message.content || 'I couldn\'t generate a response.';
        
        // Determine if response used knowledge base
        const usedKnowledgeBase = relevantSections.length > 0 && 
          relevantSections.some(section => 
            responseContent.toLowerCase().includes(section.content.toLowerCase().slice(0, 30))
          );

        assistantMessage = {
          role: 'assistant',
          content: responseContent,
          source: usedKnowledgeBase ? 'knowledge-base' : 'ai-general'
        };
      }
      
      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: "Sorry, I'm having trouble responding right now. Please try again later.",
        source: 'error'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderHome = () => (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-pink-100 to-purple-100 p-6 rounded-2xl">
        <h2 className="text-2xl font-bold mb-2">Welcome to Your Pregnancy Journey</h2>
        <p className="text-gray-700">Currently tracking: Week {currentWeek}</p>
        {dueDate && (
          <p className="text-sm text-gray-600 mt-1">
            Due Date: {new Date(dueDate).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
          </p>
        )}
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Baby className="w-8 h-8 text-purple-600" />
            <div>
              <p className="font-semibold">{weekInfo?.title || 'Your Pregnancy Progress'}</p>
              <p className="text-sm text-gray-600">{weekInfo?.trimester || 'First'} Trimester</p>
            </div>
          </div>
          <button
            onClick={() => setModalVersion(prev => prev + 1)}
            className="px-3 py-1 text-sm text-purple-600 hover:text-purple-800 underline cursor-pointer"
            type="button"
          >
            Update Due Date
          </button>
        </div>
      </div>

      {!isKbLoaded && (
        <div className="bg-yellow-50 p-4 rounded-xl border border-yellow-200">
          <p className="text-sm text-yellow-800">Loading pregnancy knowledge base...</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={() => setActiveTab('tracker')}
          className="bg-white p-4 rounded-xl shadow-md hover:shadow-lg transition-shadow"
        >
          <Calendar className="w-8 h-8 text-blue-500 mb-2" />
          <h3 className="font-semibold">Week Tracker</h3>
          <p className="text-sm text-gray-600">Track your progress</p>
        </button>

        <button
          onClick={() => setActiveTab('medications')}
          className="bg-white p-4 rounded-xl shadow-md hover:shadow-lg transition-shadow"
        >
          <Pill className="w-8 h-8 text-green-500 mb-2" />
          <h3 className="font-semibold">Medications</h3>
          <p className="text-sm text-gray-600">Check safety</p>
        </button>

        <button
          onClick={() => setActiveTab('symptoms')}
          className="bg-white p-4 rounded-xl shadow-md hover:shadow-lg transition-shadow"
        >
          <Activity className="w-8 h-8 text-orange-500 mb-2" />
          <h3 className="font-semibold">Symptoms</h3>
          <p className="text-sm text-gray-600">Track & understand</p>
        </button>

        <button
          onClick={() => setActiveTab('nutrition')}
          className="bg-white p-4 rounded-xl shadow-md hover:shadow-lg transition-shadow"
        >
          <Apple className="w-8 h-8 text-red-500 mb-2" />
          <h3 className="font-semibold">Nutrition</h3>
          <p className="text-sm text-gray-600">Daily requirements</p>
        </button>
      </div>

      <div className="bg-red-50 p-4 rounded-xl border border-red-200">
        <h3 className="font-semibold text-red-800 mb-2 flex items-center">
          <AlertTriangle className="w-5 h-5 mr-2" />
          Quick Emergency Reference
        </h3>
        <p className="text-sm text-red-700 mb-2">Seek immediate help for:</p>
        <ul className="text-sm space-y-1">
          {emergencySymptoms.slice(0, 3).map((s, i) => (
            <li key={i} className="flex items-start">
              <span className="text-red-500 mr-2">•</span>
              <span>{s.sign}</span>
            </li>
          ))}
        </ul>
        <button 
          onClick={() => setActiveTab('emergency')}
          className="text-red-600 text-sm mt-2 underline"
        >
          View all emergency symptoms →
        </button>
      </div>
    </div>
  );

  const renderTracker = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-xl shadow-md">
        <h2 className="text-2xl font-bold mb-4">Pregnancy Week Tracker</h2>
        
        <div className="mb-6">
          <div className="bg-purple-50 p-4 rounded-lg mb-4">
            <p className="text-purple-900 font-medium">
              Based on your due date: {dueDate ? new Date(dueDate).toLocaleDateString() : 'Not set'}
            </p>
            <p className="text-2xl font-bold text-purple-600 mt-1">
              You are currently in Week {currentWeek}
            </p>
          </div>
        </div>

        {weekInfo ? (
          <div className="space-y-4">
            <div className="bg-purple-50 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-900">{weekInfo.title}</h3>
              <p className="text-purple-700">{weekInfo.trimester} Trimester</p>
            </div>

            <div>
              <h4 className="font-semibold mb-2">Common Symptoms</h4>
              <div className="space-y-2">
                {weekInfo.commonSymptoms.map((symptom, i) => (
                  <div key={i} className="flex items-start bg-gray-50 p-3 rounded-lg">
                    <Heart className="w-5 h-5 text-pink-500 mr-2 mt-0.5" />
                    <div>
                      <p className="font-medium">{symptom.symptom}</p>
                      <p className="text-sm text-gray-600">{symptom.status}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {weekInfo.exercise && (
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2">
                  Recommended Exercise: {weekInfo.exercise.name}
                </h4>
                <p className="text-sm text-blue-700 mb-2">{weekInfo.exercise.benefits}</p>
                <ul className="text-sm space-y-1">
                  {weekInfo.exercise.instructions.map((inst, i) => (
                    <li key={i} className="flex items-start">
                      <span className="text-blue-500 mr-2">•</span>
                      {inst}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <Baby className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <p>Detailed information for week {currentWeek} coming soon!</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderMedications = () => {
    const searchResults = medicationSearch ? kb.checkMedicationSafety(medicationSearch) : [];
    const knowledgeBase = kb.getKnowledgeBase();
    
    return (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-xl shadow-md">
          <h2 className="text-2xl font-bold mb-4">Medication Safety Checker</h2>
          
          <div className="mb-6">
            <div className="relative">
              <Search className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search medication (e.g., Tylenol, Advil)"
                value={medicationSearch}
                onChange={(e) => setMedicationSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
          </div>

          {searchResults.length > 0 && (
            <div className="space-y-3">
              {searchResults.map((med, i) => (
                <div key={i} className={`p-4 rounded-lg border ${
                  med.safety === '🟢' ? 'bg-green-50 border-green-200' :
                  med.safety === '🟡' ? 'bg-yellow-50 border-yellow-200' :
                  'bg-red-50 border-red-200'
                }`}>
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-semibold">{med.drug}</h4>
                      <p className="text-sm text-gray-600">Brand: {med.brand || 'Generic'}</p>
                      <p className="text-sm text-gray-600">For: {med.condition}</p>
                      <p className="mt-2 font-medium flex items-center">
                        <span className="text-2xl mr-2">{med.safety}</span>
                        {med.safetyLevel}
                      </p>
                      {med.note && (
                        <p className="text-sm mt-1 text-gray-700">Note: {med.note}</p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {knowledgeBase && (
            <div className="mt-8">
              <h3 className="font-semibold mb-3">Common Medications by Category</h3>
              {knowledgeBase.medications.byCondition.map((cat: any, i: number) => (
                <div key={i} className="mb-4">
                  <h4 className="font-medium text-gray-700 mb-2">{cat.condition}</h4>
                  <div className="grid grid-cols-1 gap-2">
                    {cat.medications.map((med: any, j: number) => (
                      <div key={j} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                        <span className="text-sm">{med.drug} ({med.brand})</span>
                        <span className="text-xl">{med.safety}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderSymptoms = () => {
    const symptomResults = symptomSearch ? kb.getSymptomInfo(symptomSearch) : [];
    const knowledgeBase = kb.getKnowledgeBase();
    
    return (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-xl shadow-md">
          <h2 className="text-2xl font-bold mb-4">Symptom Tracker</h2>
          
          <div className="mb-6">
            <div className="relative">
              <Search className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Describe your symptom (e.g., bleeding, pain, fever)"
                value={symptomSearch}
                onChange={(e) => setSymptomSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
          </div>

          {symptomResults.length > 0 && (
            <div className="space-y-3 mb-6">
              {symptomResults.map((symptom, i) => (
                <div key={i} className={`p-4 rounded-lg border ${
                  symptom.severity === 'high' ? 'bg-red-50 border-red-200' :
                  symptom.severity === 'medium' ? 'bg-yellow-50 border-yellow-200' :
                  'bg-green-50 border-green-200'
                }`}>
                  <h4 className="font-semibold">{symptom.sign}</h4>
                  <p className="text-sm text-gray-600">Category: {symptom.category}</p>
                  <p className="mt-2">
                    <span className="font-medium">Action:</span> {symptom.action}
                  </p>
                  <p className="text-sm">
                    <span className="font-medium">Contact provider:</span> {symptom.urgency}
                  </p>
                </div>
              ))}
            </div>
          )}

          {knowledgeBase && (
            <div>
              <h3 className="font-semibold mb-3">All Symptoms by Category</h3>
              {knowledgeBase.symptomTroubleshooting.categories.map((cat: any, i: number) => (
                <div key={i} className="mb-6">
                  <h4 className="font-medium text-gray-700 mb-2">{cat.category}</h4>
                  <div className="space-y-2">
                    {cat.symptoms.map((symptom: any, j: number) => (
                      <div key={j} className="flex items-start p-3 bg-gray-50 rounded-lg">
                        <div className={`w-2 h-2 rounded-full mt-1.5 mr-3 ${
                          symptom.severity === 'high' ? 'bg-red-500' :
                          symptom.severity === 'medium' ? 'bg-yellow-500' :
                          'bg-green-500'
                        }`} />
                        <div className="flex-1">
                          <p className="text-sm font-medium">{symptom.sign}</p>
                          <p className="text-xs text-gray-600">{symptom.action} • {symptom.urgency}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderNutrition = () => {
    const knowledgeBase = kb.getKnowledgeBase();
    
    return (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-xl shadow-md">
          <h2 className="text-2xl font-bold mb-4">Nutrition Planner</h2>
          
          <div className="mb-6">
            <h3 className="font-semibold mb-3">Daily Nutritional Requirements</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {nutritionalReqs.dailyMacros.map((nutrient, i) => (
                <div key={i} className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg">
                  <div className="flex justify-between items-center">
                    <div>
                      <h4 className="font-semibold">{nutrient.nutrient}</h4>
                      <p className="text-sm text-gray-600 capitalize">{nutrient.category}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-lg font-bold text-purple-600">{nutrient.amount}</p>
                      <p className="text-sm text-gray-600">{nutrient.unit}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {knowledgeBase && (
            <>
              <div className="bg-red-50 p-4 rounded-lg mb-6">
                <h3 className="font-semibold text-red-900 mb-2">Foods to Avoid</h3>
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium text-red-800 mb-1">Unsafe Seafood (High Mercury)</h4>
                    <p className="text-sm text-red-700">
                      {knowledgeBase.foodSafety.seafoodGuidelines.unsafe.join(', ')}
                    </p>
                  </div>
                  <div className="space-y-2">
                    {knowledgeBase.foodSafety.avoidFoods.map((food: any, i: number) => (
                      <div key={i} className="text-sm">
                        <span className="font-medium text-red-800">{food.item}</span>
                        {food.includes && (
                          <span className="text-red-700"> - includes {food.includes.join(', ')}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="font-semibold text-green-900 mb-2">Morning Sickness Tips</h3>
                <div className="space-y-2 text-sm">
                  <div>
                    <p className="font-medium text-green-800">What to Eat:</p>
                    <p className="text-green-700">{knowledgeBase.morningSicknessManagement.whatToEat.join(', ')}</p>
                  </div>
                  <div>
                    <p className="font-medium text-green-800">Foods to Avoid:</p>
                    <p className="text-green-700">{knowledgeBase.morningSicknessManagement.avoidFoods.join(', ')}</p>
                  </div>
                  <div>
                    <p className="font-medium text-green-800">Eating Tips:</p>
                    <ul className="text-green-700 ml-4">
                      {knowledgeBase.morningSicknessManagement.eatingTips.map((tip: string, i: number) => (
                        <li key={i} className="list-disc">{tip}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    );
  };

  const renderEmergency = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-xl shadow-md">
        <h2 className="text-2xl font-bold mb-4 text-red-600 flex items-center">
          <AlertTriangle className="w-8 h-8 mr-2" />
          Emergency Symptoms Guide
        </h2>
        
        <div className="bg-red-50 p-4 rounded-lg mb-6 border border-red-200">
          <p className="text-red-800 font-semibold mb-2">
            Seek immediate medical attention for any of these symptoms:
          </p>
        </div>

        <div className="space-y-3">
          {emergencySymptoms.map((symptom, i) => (
            <div key={i} className="p-4 bg-red-50 rounded-lg border border-red-200">
              <div className="flex items-start">
                <AlertCircle className="w-6 h-6 text-red-600 mr-3 mt-0.5" />
                <div className="flex-1">
                  <h4 className="font-semibold text-red-900">{symptom.sign}</h4>
                  <p className="text-sm text-red-700 mt-1">Category: {symptom.category}</p>
                  <p className="text-sm text-red-700">Action: {symptom.action}</p>
                  <p className="font-medium text-red-800 mt-2">
                    Contact provider: {symptom.urgency}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <h3 className="font-semibold text-blue-900 mb-2">Emergency Contacts</h3>
          <p className="text-sm text-blue-800">Keep these numbers handy:</p>
          <ul className="mt-2 space-y-1 text-sm">
            <li>• Your OB/GYN: _______________</li>
            <li>• Hospital L&D: _______________</li>
            <li>• Emergency: 911</li>
          </ul>
        </div>
      </div>
    </div>
  );

  const renderChat = () => (
    <div className="flex flex-col h-full bg-white rounded-xl shadow-md">
      <div className="p-4 border-b">
        <h2 className="text-xl font-bold flex items-center">
          <MessageCircle className="w-6 h-6 mr-2" />
          Pregnancy Assistant
        </h2>
        <p className="text-xs text-gray-500 mt-1">Powered by OpenAI GPT-3.5</p>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chatMessages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[70%] p-3 rounded-lg ${
              msg.role === 'user' 
                ? 'bg-purple-500 text-white' 
                : 'bg-gray-100 text-gray-800'
            }`}>
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.source === 'knowledge-base' && (
                <p className="text-xs mt-2 text-green-600 flex items-center">
                  <CheckCircle className="w-3 h-3 mr-1" />
                  Based on verified medical information
                </p>
              )}
              {msg.source === 'ai-general' && (
                <p className="text-xs mt-2 text-yellow-600 flex items-center">
                  <AlertCircle className="w-3 h-3 mr-1" />
                  General advice - consult your doctor
                </p>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[70%] p-3 rounded-lg bg-gray-100 text-gray-800">
              <div className="flex space-x-2">
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
      
      <form onSubmit={handleChatSubmit} className="p-4 border-t">
        <div className="flex space-x-2">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder="Ask about symptoms, medications, or pregnancy..."
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors disabled:opacity-50"
            disabled={isLoading || !chatInput.trim()}
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          AI responses use your pregnancy knowledge base when relevant. Always consult your healthcare provider.
        </p>
      </form>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-4">
        {/* Header */}
        <header className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800 flex items-center">
            <Baby className="w-8 h-8 mr-2 text-purple-600" />
            Pregnancy Care Companion
          </h1>
          <p className="text-gray-600 mt-1">Your complete pregnancy tracking and health guide</p>
        </header>

        {/* Main Content */}
        <div className="pb-20">
          {activeTab === 'home' && renderHome()}
          {activeTab === 'tracker' && renderTracker()}
          {activeTab === 'medications' && renderMedications()}
          {activeTab === 'symptoms' && renderSymptoms()}
          {activeTab === 'nutrition' && renderNutrition()}
          {activeTab === 'emergency' && renderEmergency()}
          {activeTab === 'chat' && (
            <div className="h-[600px]">
              {renderChat()}
            </div>
          )}
        </div>

        {/* Due Date Modal */}
        {showDueDateModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-xl max-w-md w-full">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold">Update Due Date</h3>
                <button 
                  onClick={() => setModalVersion(0)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
              <form onSubmit={handleDueDateSubmit}>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Enter your due date:
                  </label>
                  <input
                    type="date"
                    name="dueDate"
                    required
                    className="w-full p-2 border rounded-lg"
                    min={new Date().toISOString().split('T')[0]}
                  />
                </div>
                <button
                  type="submit"
                  className="w-full bg-purple-600 text-white py-2 rounded-lg hover:bg-purple-700"
                >
                  Save Due Date
                </button>
              </form>
            </div>
          </div>
        )}

        {/* Bottom Navigation */}
        <nav className="fixed bottom-0 left-0 right-0 bg-white border-t shadow-lg">
          <div className="max-w-4xl mx-auto px-4">
            <div className="flex justify-around py-2">
              <button
                onClick={() => setActiveTab('home')}
                className={`flex flex-col items-center p-2 ${
                  activeTab === 'home' ? 'text-purple-600' : 'text-gray-500'
                }`}
              >
                <Home className="w-6 h-6" />
                <span className="text-xs mt-1">Home</span>
              </button>
              
              <button
                onClick={() => setActiveTab('tracker')}
                className={`flex flex-col items-center p-2 ${
                  activeTab === 'tracker' ? 'text-purple-600' : 'text-gray-500'
                }`}
              >
                <Calendar className="w-6 h-6" />
                <span className="text-xs mt-1">Tracker</span>
              </button>
              
              <button
                onClick={() => setActiveTab('medications')}
                className={`flex flex-col items-center p-2 ${
                  activeTab === 'medications' ? 'text-purple-600' : 'text-gray-500'
                }`}
              >
                <Pill className="w-6 h-6" />
                <span className="text-xs mt-1">Meds</span>
              </button>
              
              <button
                onClick={() => setActiveTab('emergency')}
                className={`flex flex-col items-center p-2 ${
                  activeTab === 'emergency' ? 'text-purple-600' : 'text-gray-500'
                }`}
              >
                <AlertCircle className="w-6 h-6" />
                <span className="text-xs mt-1">Emergency</span>
              </button>
              
              <button
                onClick={() => setActiveTab('chat')}
                className={`flex flex-col items-center p-2 ${
                  activeTab === 'chat' ? 'text-purple-600' : 'text-gray-500'
                }`}
              >
                <MessageCircle className="w-6 h-6" />
                <span className="text-xs mt-1">Chat</span>
              </button>
            </div>
          </div>
        </nav>
      </div>
    </div>
  );
};

export default PregnancyTrackerApp;